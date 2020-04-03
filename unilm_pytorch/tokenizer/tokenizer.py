from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences



class Tokenizer:
    """ Tokenizer class"""

    def __init__(self, split_fn, add_tokens=False):
        
        if add_tokens:
            special_tokens_dict = {'bos_token': '[BOS]', 'eos_token':'[EOS]'}
            num_added_toks = split_fn.add_special_tokens(special_tokens_dict)
        self.tokenizer = split_fn
        self.cls_idx = self.tokenizer.cls_token_id # vocab.to_indices('[CLS]')
        self.sep_idx = self.tokenizer.sep_token_id # vocab.to_indices('[SEP]')
        self.bos_idx = self.tokenizer.bos_token_id # vocab.to_indices('[BOS]')
        self.eos_idx = self.tokenizer.eos_token_id # vocab.to_indices('[EOS]')
        self.mask_idx = self.tokenizer.mask_token_id   # vocab.to_indices('[MASK]')
        self.pad_idx = self.tokenizer.pad_token_id     # vocab.to_indices('[PAD]')
        self.unk_idx = self.tokenizer.unk_token_id
        self.vocab_size = len(self.tokenizer)
 
        
    def __call__(self, text_string):
        return self.tokenizer.tokenize(text_string)
    
    def __len__(self):
        return len(self.tokenizer)
 

    def idx2token2(self, idx):
            
        if idx not in self.tokenizer.idx2token.keys():
            return '[UNK]'
        else:
            return self.tokenizer.idx2token[idx]
        
        
    def token2idx(self, token):
            
        if token not in self.tokenizer.token2idx.keys():
            return self.unk_idx
        else:
            return self.tokenizer.token2idx[token]
    
    
    def token_to_idx(self, text_list):
        
        idx_tok = []
        for t in text_list:
            if t not in self.tokenizer.token2idx.keys():
                idx_tok.append(self.unk_idx)
            else:
                idx = self.tokenizer.token2idx[t]
                idx_tok.append(idx)

        return idx_tok
    
    
    def token_to_idx_with_bos(self, text_list):      
        idx_tok = self.token_to_idx(text_list)
        idx_tok = [self.bos_idx] + idx_tok 
        
        return idx_tok
    
    
    def token_to_idx_with_eos(self, text_list):
        idx_tok = self.token_to_idx(text_list)
        idx_tok = idx_tok + [self.eos_idx]  
        return idx_tok
    
    
    def token_to_cls_sep_idx(self, text_list):
        
        # tokenized_text_list = sentencepiece_tokenizer(text)
        idx_tok = self.token_to_idx(text_list)
        idx_tok = [self.cls_idx] + idx_tok + [self.sep_idx]

        return idx_tok
    
    
    def text_to_idx_with_bos(self, text):  
        tokenized_list = self.tokenizer.tokenize(text)
        idx_tok = self.token_to_idx(tokenized_list)
        idx_tok = [self.bos_idx] + idx_tok 
        
        return idx_tok
    
    
    def text_to_idx_with_eos(self, text):
        tokenized_list = self.tokenizer.tokenize(text)
        idx_tok = self.token_to_idx(tokenized_list)
        idx_tok = idx_tok + [self.eos_idx]  
        return idx_tok
    
    
    def text_to_cls_sep_idx(self, text):
        tokenized_list = self.tokenizer.tokenize(text)
        
        # tokenized_text_list = sentencepiece_tokenizer(text)
        idx_tok = self.token_to_idx(tokenized_list)
        idx_tok = [self.cls_idx] + idx_tok + [self.sep_idx]

        return idx_tok
    
    
    def text_to_sep_idx(self, text):
        tokenized_list = self.tokenizer.tokenize(text)
        
        # tokenized_text_list = sentencepiece_tokenizer(text)
        idx_tok = self.token_to_idx(tokenized_list)
        idx_tok = idx_tok + [self.sep_idx]

        return idx_tok
    
    
    def idx_to_token(self, idx_list):
        out = []
        for i in idx_list:
            token = self.tokenizer.idx2token[i]
            out.append(token)
        
        return out
    
    def pad(self, sequence, maxlen):
        if len(sequence)>=maxlen:
            return sequence[:maxlen]
        else:
            extra_len = maxlen-len(sequence)
            sequence = sequence + [self.pad_idx]*extra_len
            
            return sequence
    
    