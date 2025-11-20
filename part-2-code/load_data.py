import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_queries = load_lines(nl_path)
        
        data = []
        if split != 'test':
            sql_path = os.path.join(data_folder, f'{split}.sql')
            sql_queries = load_lines(sql_path)
            for nl, sql in zip(nl_queries, sql_queries):
                data.append({'nl': nl, 'sql': sql})
        else:
            for nl in nl_queries:
                data.append({'nl': nl})
        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoder_input = self.tokenizer(item['nl'], return_tensors='pt', truncation=True, max_length=1024)
        
        if self.split != 'test':
            decoder_target = self.tokenizer(item['sql'], return_tensors='pt', truncation=True, max_length=1024)
            return {
                'encoder_ids': encoder_input['input_ids'].squeeze(0),
                'encoder_mask': encoder_input['attention_mask'].squeeze(0),
                'decoder_ids': decoder_target['input_ids'].squeeze(0)
            }
        else:
            return {
                'encoder_ids': encoder_input['input_ids'].squeeze(0),
                'encoder_mask': encoder_input['attention_mask'].squeeze(0)
            }

def normal_collate_fn(batch):
    encoder_ids = pad_sequence([item['encoder_ids'] for item in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence([item['encoder_mask'] for item in batch], batch_first=True, padding_value=0)
    decoder_ids = pad_sequence([item['decoder_ids'] for item in batch], batch_first=True, padding_value=PAD_IDX)
    
    # T5 expects decoder_input_ids to be shifted right with pad token at start
    decoder_inputs = torch.cat([torch.zeros((decoder_ids.shape[0], 1), dtype=torch.long), decoder_ids[:, :-1]], dim=1)
    decoder_targets = decoder_ids
    initial_decoder_inputs = torch.zeros((len(batch), 1), dtype=torch.long)
    
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    encoder_ids = pad_sequence([item['encoder_ids'] for item in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence([item['encoder_mask'] for item in batch], batch_first=True, padding_value=0)
    
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    initial_decoder_inputs = torch.full((len(batch), 1), tokenizer.pad_token_id, dtype=torch.long)
    
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    return train_x, train_y, dev_x, dev_y, test_x