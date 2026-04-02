import math
import json
import torch
import random
import datetime
from rouge import rouge
from bleu import compute_bleu

def rouge_score(references, generated):
    """Calculate ROUGE scores for a list of strings."""
    score = rouge(generated, references)
    return {k: (v * 100) for k, v in score.items()}

def bleu_score(references, generated, n_gram=4, smooth=False):
    """Calculate BLEU score for a list of lists of tokens."""
    formatted_ref = [[ref] for ref in references]
    bleu_s, *_ = compute_bleu(formatted_ref, generated, n_gram, smooth)
    return bleu_s * 100

class ExpDataLoader:
    def __init__(self, data_dir):
        with open(data_dir + 'explanation.json', 'r') as f:
            self.exp_data = json.load(f)

        self.train = self.exp_data['train']
        self.valid = self.exp_data['val']
        self.test = self.exp_data['test']

class SeqDataLoader:
    def __init__(self, data_dir):
        self.user2items_positive = {}
        with open(data_dir + 'sequential.txt', 'r') as f:
            for line in f:
                user, items = line.strip().split(' ', 1)
                self.user2items_positive[int(user)] = items.split(' ')

        self.user2items_negative = {}
        with open(data_dir + 'negative.txt', 'r') as f:
            for line in f:
                user, items = line.strip().split(' ', 1)
                self.user2items_negative[int(user)] = items.split(' ')

        with open(data_dir + 'datamaps.json', 'r') as f:
            datamaps = json.load(f)
            
        self.id2user = datamaps['user2id']
        self.id2item = datamaps['item2id']

class ExpSampler:
    def __init__(self, exp_data):
        self.task_id = 0
        self.exp_data = exp_data
        self.sample_num = len(self.exp_data)
        self.index_list = list(range(self.sample_num))
        self.step = 0

    def check_step(self):
        if self.step == self.sample_num:
            self.step = 0
            random.shuffle(self.index_list)  # Shuffle indices at epoch start for better generalization

    def sample(self, num):
        # Sample a batch of data, returning task ids, user/item inputs, and targets
        task = [self.task_id] * num
        user_inputs, item_inputs, outputs = [], [], []
        for _ in range(num):
            self.check_step()
            idx = self.index_list[self.step]
            record = self.exp_data[idx]
            user_inputs.append(f"user_{record['user']}")
            item_inputs.append(f"item_{record['item']}")
            outputs.append(record['explanation'])
            self.step += 1
        return task, user_inputs, item_inputs, outputs

class SeqSampler:
    def __init__(self, user2items_pos):
        self.task_id = 1
        self.max_seq_len = 21
        self.item_template = ' item_'

        self.user2items_pos = user2items_pos
        self.user_list = list(user2items_pos.keys())

        self.sample_num = len(self.user_list)
        self.index_list = list(range(self.sample_num))
        self.step = 0

    def check_step(self):
        if self.step == self.sample_num:
            self.step = 0
            random.shuffle(self.index_list)

    def sample_seq(self, u):