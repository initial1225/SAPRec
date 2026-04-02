import os
import torch
import random
import argparse
from transformers import T5Tokenizer
from utils import SeqDataLoader, TopNBatchify, now_time, evaluate_ndcg, evaluate_hr, evaluate_recall, evaluate_mrr

parser = argparse.ArgumentParser(description='Top-N Recommendation Evaluation')
parser.add_argument('--data_dir', type=str, default='./data/beauty', help='Directory for loading data')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--cuda', action='store_true', help='Use CUDA')
parser.add_argument('--checkpoint', type=str, default='./checkpoints/beauty/', help='Directory containing the saved model')
parser.add_argument('--negative_num', type=int, default=99, help='Number of negative items for top-N recommendation')
parser.add_argument('--num_beams', type=int, default=20, help='Number of beams for beam search')
parser.add_argument('--top_n', type=int, default=10, help='Number of items to predict per user')
args = parser.parse_args()

model_version = 't5-small'

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print(f'{arg:40} {getattr(args, arg)}')
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

# Configure device
device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
if args.cuda and not torch.cuda.is_available():
    print(f"{now_time()} WARNING: CUDA requested but not available. Using CPU.")

model_path = os.path.join(args.checkpoint, 'model.pt')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model checkpoint not found at {model_path}")


# 1. Load Data
print(f"{now_time()} Loading data...")
tokenizer = T5Tokenizer.from_pretrained(model_version)
seq_corpus = SeqDataLoader(args.data_dir)
nitem = len(seq_corpus.id2item)
topn_iterator = TopNBatchify(seq_corpus.user2items_positive, seq_corpus.user2items_negative, args.negative_num, nitem, tokenizer, args.batch_size)


# 2. Generation Loop
def generate(model):
    model.eval()
    idss_predict = []
    
    with torch.no_grad():
        while True:
            # Fetch batch and cleanly move all tensors to the target device
            batch_data = topn_iterator.next_batch_test()
            task, source_user, source_item, source_user_mask, source_item_mask, whole_word_user, whole_word_item, _ = \
                [x.to(device) for x in batch_data]

            # Generate recommendations using beam search
            beam_outputs = model.beam_search(
                task, source_user, source_item, whole_word_user, whole_word_item, source_user_mask, source_item_mask,
                num_beams=args.num_beams,
                num_return_sequences=args.top_n
            )

            # Decode outputs
            output_tensor = beam_outputs.view(task.size(0), args.top_n, -1)
            for i in range(task.size(0)):
                results = tokenizer.batch_decode(output_tensor[i], skip_special_tokens=True)
                idss_predict.append(results)

            if topn_iterator.step == topn_iterator.total_step:
                break
                
    return idss_predict

# Load the full model object
with open(model_path, 'rb') as f:
    model = torch.load(f, weights_only=False).to(device)

print(f"{now_time()} Generating recommendations...")
idss_predicted = generate(model)

print(f"{now_time()} Evaluation...")

# Build ground truth dictionary (last item in sequence)
user2item_test = {
    user: [int(item_list[-1])]
    for user, item_list in seq_corpus.user2items_positive.items()
}

# Build prediction dictionary
user2rank_list = {}
for predictions, user in zip(idss_predicted, topn_iterator.user_list):
    prediction_list = []
    for p in predictions:
        try:
            prediction_list.append(int(p.split(' ')[0]))  # Extract ID before whitespace
        except (ValueError, IndexError):
            prediction_list.append(random.randint(1, nitem))  # Fallback to random recommendation
    user2rank_list[user] = prediction_list

# Dynamically define evaluation cutoffs based on args.top_n
top_ns = [k for k in [5, 10] if k <= args.top_n]

for top_n in top_ns:
    hr = evaluate_hr(user2item_test, user2rank_list, top_n)
    ndcg = evaluate_ndcg(user2item_test, user2rank_list, top_n)
    
    print(f"{now_time()} HR@{top_n:<2}     {hr:7.4f}")
    print(f"{now_time()} NDCG@{top_n:<2}   {ndcg:7.4f}")