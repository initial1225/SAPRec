import os
import torch
import argparse
from transformers import T5Tokenizer
from utils import rouge_score, bleu_score, ExpDataLoader, ExpBatchify, now_time, ids2tokens

parser = argparse.ArgumentParser(description='SAPRec Explanation Generation Evaluation')
parser.add_argument('--data_dir', type=str, default='./data/beauty/', help='Directory for loading data')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--cuda', action='store_true', help='Use CUDA')
parser.add_argument('--checkpoint', type=str, default='./checkpoint/beauty/', help='Directory to load the final model')
parser.add_argument('--outf', type=str, default='./output/beauty/generated.txt', help='Output file for generated text')
parser.add_argument('--num_beams', type=int, default=21, help='Number of beams')
parser.add_argument('--num_beam_groups', type=int, default=3, help='Number of beam groups')
parser.add_argument('--min_len', type=int, default=10, help='Minimum length of an explanation')
parser.add_argument('--exp_len', type=int, default=20, help='Maximum length of an explanation')
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

# Ensure output directory exists for the generated text file
prediction_path = args.outf
os.makedirs(os.path.dirname(prediction_path), exist_ok=True)

# 1. Load Data
print(f"{now_time()} Loading data...")
tokenizer = T5Tokenizer.from_pretrained(model_version)
exp_corpus = ExpDataLoader(args.data_dir)
exp_iterator = ExpBatchify(exp_corpus.test, tokenizer, args.exp_len, args.batch_size)

# 2. Generation Loop
def generate(model):
    model.eval()
    idss_predict = []
    
    with torch.no_grad():
        while True:
            # Fetch batch and move tensors to device cleanly
            batch_data = exp_iterator.next_batch_test()
            task, source_user, source_item, source_user_mask, source_item_mask, whole_word_user, whole_word_item, _ = \
                [x.to(device) for x in batch_data]

            # Generate explanations using beam search
            beam_outputs = model.beam_search(
                task, source_user, source_item, whole_word_user, whole_word_item, source_user_mask, source_item_mask,
                min_length=args.min_len,
                num_beams=args.num_beams,
                num_beam_groups=args.num_beam_groups,
                num_return_sequences=1
            )

            idss_predict.extend(beam_outputs.tolist())

            if exp_iterator.step == exp_iterator.total_step:
                break
                
    return idss_predict


# Load the full model object
with open(model_path, 'rb') as f:
    model = torch.load(f, weights_only=False).to(device)

print(f"{now_time()} Generating text...")
idss_predicted = generate(model)

print(f"{now_time()} Evaluation...")

# Decode tokens
tokens_test = [ids2tokens(ids, tokenizer) for ids in exp_iterator.target_seq.tolist()]
tokens_predict = [ids2tokens(ids, tokenizer) for ids in idss_predicted]

# Calculate and print BLEU scores
BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
print(f"{now_time()} BLEU-1 {BLEU1:7.4f}")

BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
print(f"{now_time()} BLEU-4 {BLEU4:7.4f}")

# Calculate and print ROUGE scores
text_test = [' '.join(tokens) for tokens in tokens_test]
text_predict = [' '.join(tokens) for tokens in tokens_predict]

ROUGE = rouge_score(text_test, text_predict)
for k, v in ROUGE.items():
    print(f"{now_time()} {k:<15} {v:7.4f}")

# Save generated texts alongside ground truth using efficient string joining
text_out = "".join(
    f"Real: {real}\nFake: {fake}\n\n" 
    for real, fake in zip(text_test, text_predict)
)

with open(prediction_path, 'w', encoding='utf-8') as f:
    f.write(text_out)

print(f"{now_time()} Generated text saved to ({prediction_path})")