import os
import torch
import torch.nn.functional as F
import argparse
import shutil
import random
import numpy as np
from transformers import T5Tokenizer
from module import SapModel
from utils import ExpDataLoader, SeqDataLoader, TrainBatchify, ExpBatchify, SeqBatchify, TopNBatchify, now_time

parser = argparse.ArgumentParser(description='SAPRec:')
parser.add_argument('--data_dir', type=str, default='./data/sports/', help='directory for loading the data')
parser.add_argument('--task_num', type=int, default=3, help='task number')
parser.add_argument('--seg_templates_len', type=int, default=5, help='seg_templates_len per task')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lambda_scale', type=float, default=0.1, help='lambda_scale')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200, help='report interval')
parser.add_argument('--checkpoint', type=str, default='./checkpoints/sports3407/ssp-3-5', help='directory to save the final model')
parser.add_argument('--endure_times', type=int, default=5, help='the maximum endure times of loss increasing on validation')
parser.add_argument('--exp_len', type=int, default=20, help='the maximum length of an explanation')
parser.add_argument('--negative_num', type=int, default=99, help='number of negative items for top-n recommendation')
args = parser.parse_args()

model_version = 't5-small'

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

if torch.cuda.is_available() and not args.cuda:
    print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')

device = torch.device('cuda' if args.cuda else 'cpu')

# Recreate checkpoint directory
if os.path.exists(args.checkpoint):
    shutil.rmtree(args.checkpoint)
os.makedirs(args.checkpoint)
model_path = os.path.join(args.checkpoint, 'model.pt')

# Data Loading
print(now_time() + 'load data ......')
tokenizer = T5Tokenizer.from_pretrained(model_version)

# Load explanation and sequential data
exp_corpus = ExpDataLoader(args.data_dir)
seq_corpus = SeqDataLoader(args.data_dir)

nitem = len(seq_corpus.id2item)

# Initialize batch iterators
all_iterator = TrainBatchify(exp_corpus.train, seq_corpus.user2items_positive, args.negative_num, nitem, tokenizer, args.exp_len, args.batch_size)
exp_iterator = ExpBatchify(exp_corpus.valid, tokenizer, args.exp_len, args.batch_size)
seq_iterator = SeqBatchify(seq_corpus.user2items_positive, tokenizer, args.batch_size)
topn_iterator = TopNBatchify(seq_corpus.user2items_positive, seq_corpus.user2items_negative, args.negative_num, nitem, tokenizer, args.batch_size)

# Init SapModel from t5-small
model = SapModel.from_pretrained(model_version)
model.init_templates(args.task_num, args.seg_templates_len, device)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

def train():
    model.train()
    text_loss = 0.
    total_sample = 0

    # Moving average of task attentions (IDs: 0=Seq, 1=TopN, 2=Exp)
    task_attention_history = {0: 0.1, 1: 0.1, 2: 0.1} 
    
    # Hyperparameters
    momentum = 0.9 
    temperature = 0.5 
    lambda_scale = args.lambda_scale 
    N = 3 

    while True:
        # Fetch and move batch to device
        batch_data = all_iterator.next_batch()
        task, source_user, source_item, source_user_mask, source_item_mask, whole_word_user, whole_word_item, target = \
            [x.to(device) for x in batch_data]

        # Forward pass (set_to_none=True optimizes memory)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(task, source_user, source_item, whole_word_user, whole_word_item, source_user_mask, source_item_mask, labels=target)
        loss = outputs.loss

        # A. Get current task ID
        current_task_id = task[0].item()

        # B. Focus on Task Prompt (P3) cross-attentions
        last_layer_attn = outputs.cross_attentions[-1]
        p3_start = args.seg_templates_len * 1
        p3_attn = last_layer_attn[:, :, :, -p3_start:]
        
        # Calculate mean attention score
        current_score = p3_attn.mean(dim=1).mean(dim=1).sum(dim=-1).mean().item()

        # C. Momentum Update
        task_attention_history[current_task_id] = (
            momentum * task_attention_history[current_task_id] + 
            (1 - momentum) * current_score
        )

        # D. Vector Level Normalize & Softmax
        scores_list = [task_attention_history[t_id] for t_id in sorted(task_attention_history.keys())]
        scores_tensor = torch.tensor(scores_list, device=loss.device)
        softmax_scores = F.softmax(scores_tensor / temperature, dim=0)

        # E. Map to weights
        all_tasks_weights = 1.0 - lambda_scale * (softmax_scores * N - 1.0)

        # F. Clamp for stability and normalize sum to N (3.0)
        all_tasks_weights = torch.clamp(all_tasks_weights, 1-lambda_scale, 1+lambda_scale)
        all_tasks_weights = all_tasks_weights / all_tasks_weights.sum() * 3.0

        # G. Apply weight
        current_batch_weight = all_tasks_weights[current_task_id]
        loss = loss * current_batch_weight

        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Logging
        batch_size = task.size(0)
        text_loss += batch_size * loss.item()
        total_sample += batch_size

        if all_iterator.batch_index % args.log_interval == 0 or all_iterator.batch_index % all_iterator.batch_num == 0:
            cur_t_loss = text_loss / total_sample
            print(now_time() + ' text loss {:4.4f} | {:5d}/{:5d} batches'.format(cur_t_loss, all_iterator.batch_index, all_iterator.batch_num))
            text_loss = 0.
            total_sample = 0

        if all_iterator.batch_index % all_iterator.batch_num == 0:
            break

def evaluate(iterator):
    model.eval()
    text_loss = 0.
    total_sample = 0
    with torch.no_grad():
        while True:
            batch_data = iterator.next_batch_valid()
            task, source_user, source_item, source_user_mask, source_item_mask, whole_word_user, whole_word_item, target = \
                [x.to(device) for x in batch_data]
            
            outputs = model(task, source_user, source_item, whole_word_user, whole_word_item, source_user_mask, source_item_mask, labels=target)
            loss = outputs.loss
            
            batch_size = task.size(0)
            text_loss += batch_size * loss.item()
            total_sample += batch_size

            if iterator.step == iterator.total_step:
                break
    return text_loss / total_sample

# Save initial model state
with open(model_path, 'wb') as f:
    torch.save(model.state_dict(), f)

print(now_time() + 'Start training')

best_val_loss = float('inf')
endure_count = 0

for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train()
    
    seq_loss = evaluate(seq_iterator)
    print(now_time() + 'sequential loss {:4.4f}'.format(seq_loss))
    
    topn_loss = evaluate(topn_iterator)
    print(now_time() + 'top-N loss {:4.4f}'.format(topn_loss))

    print(now_time() + 'validation')
    exp_loss = evaluate(exp_iterator)
    print(now_time() + 'explanation loss {:4.4f}'.format(exp_loss))
    
    val_loss = (topn_loss + seq_loss + exp_loss) / 3
    print(now_time() + 'total loss {:4.4f}'.format(val_loss))
    
    # Save the model if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model.state_dict(), f)
    else:
        endure_count += 1
        print(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == args.endure_times:
            print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break