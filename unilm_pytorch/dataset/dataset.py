from torch.utils.data import Dataset
import tqdm
import torch
import random


class UNILMDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                self.corpus_lines=0
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.lines = [l for l in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
#                 self.lines = [line[:-1].split("\t")
#                               for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            for _ in range(random.randint(0, self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        corpus = self.get_corpus_line(item).split('\t')
        t1 = corpus[0]
        t2 = corpus[1]
        
        # t1  = self.get_corpus_line(item)
        t1_sent, t1_label = self.random_word(t1)
        t2_sent, t2_label = self.random_word(t2)
        
        # Create input, label and mask, depending on mask scheme
        prob = random.random()
        prob2 = random.random()
        if prob< 2/6 : # Bidirectional LM
            unilm_input = [self.vocab.cls_idx] + t1_sent + [self.vocab.sep_idx] + t2_sent + [self.vocab.sep_idx]
            # unilm_label = [self.vocab.pad_idx] + t1_label + [self.vocab.pad_idx] + t2_label + [self.vocab.pad_idx]
            unilm_label = [self.vocab.unk_idx] + t1_label + [self.vocab.unk_idx] + t2_label + [self.vocab.unk_idx]
            segment_label = ([1 for _ in range(len(t1_sent)+2)] + [2 for _ in range(len(t2_sent)+1)])
            seq_mask = self.get_bidirectional_lm_mask(unilm_input)
           
        elif 2/6 < prob < 3/6: # Left-to-Right LM
            if prob2 < 0.5:
                sent, label = t1_sent, t1_label
            else: 
                sent, label = t2_sent, t2_label
            unilm_input = [self.vocab.cls_idx] + sent + [self.vocab.sep_idx]
            unilm_label = [self.vocab.unk_idx] + label + [self.vocab.unk_idx]
            # unilm_label = [self.vocab.pad_idx] + label + [self.vocab.pad_idx]
            segment_label = [1 for _ in range(len(sent)+2)]
            seq_mask = self.get_left2right_lm_mask(unilm_input)
            
        elif 3/6 < prob < 4/6: # Right-to-Left LM
            if prob2 < 0.5:
                sent, label = t1_sent, t1_label
            else: 
                sent, label = t2_sent, t2_label
            unilm_input = [self.vocab.cls_idx] + sent + [self.vocab.sep_idx]
            unilm_label = [self.vocab.unk_idx] + label + [self.vocab.unk_idx]
            # unilm_label = [self.vocab.pad_idx] + label + [self.vocab.pad_idx]
            segment_label = [1 for _ in range(len(sent)+2)]
            seq_mask = self.get_right2left_lm_mask(unilm_input)
        
        else: # Seq-to-Seq LM
            unilm_input = [self.vocab.cls_idx] + t1_sent + [self.vocab.sep_idx] + t2_sent + [self.vocab.sep_idx]
            unilm_label = [self.vocab.unk_idx] + t1_label + [self.vocab.unk_idx] + t2_label + [self.vocab.unk_idx]
            # unilm_label = [self.vocab.pad_idx] + t1_label + [self.vocab.pad_idx] + t2_label + [self.vocab.pad_idx]
            segment_label = ([1 for _ in range(len(t1_sent)+2)] + [2 for _ in range(len(t2_sent)+1)])
            seq_mask = self.get_seq2seq_lm_mask(unilm_input)
        
        assert len(unilm_input)==len(unilm_label)==len(segment_label)
        
        # Pad sequence
        if len(unilm_input) < self.seq_len:
            # padding = [self.vocab.pad_idx for _ in range(self.seq_len - len(unilm_input))]
            padding = [self.vocab.unk_idx for _ in range(self.seq_len - len(unilm_input))]
            unilm_input.extend(padding), unilm_label.extend(padding), segment_label.extend(padding)
        else:
            unilm_input = unilm_input[:self.seq_len]
            unilm_label = unilm_label[:self.seq_len]
            segment_label = segment_label[:self.seq_len]
                               
        output = {"unilm_input": unilm_input,
                  "unilm_label": unilm_label,
                  "segment_label": segment_label,
                  "unilm_mask":  seq_mask.unsqueeze(1)
                 }
        
        # return {key: value.clone().detach() for key, value in output.items()}
        return {key: torch.tensor(value) for key, value in output.items()}

    
    def random_word(self, sentence):
        tokens = self.vocab(sentence) # Tokenizing
        output_label = []
         
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_idx

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.token2idx(token)

                output_label.append(self.vocab.token2idx(token))

            else:
                tokens[i] = self.vocab.token2idx(token)
                output_label.append(self.vocab.unk_idx) 

        return tokens, output_label
    

    def get_corpus_line(self, index):
        if self.on_memory:
            return self.lines[index]
        else:
            line = self.file.__next__()
            if line is None:
                print('No use of Memory !')
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()
            return line

    def get_bidirectional_lm_mask(self,tokens):
        mask = torch.ones((self.seq_len, self.seq_len))        
        return mask
    
    
    def get_left2right_lm_mask(self,tokens):
        mask = torch.ones((self.seq_len, self.seq_len))        
        return torch.tril(mask)
    

    def get_right2left_lm_mask(self,tokens):
        mask = torch.ones((self.seq_len, self.seq_len))        
        return torch.triu(mask)
    
    
    def get_seq2seq_lm_mask(self,tokens):
        """
        Produce a maks for source and target.
        """
        for i,j in enumerate(tokens):
            if j=='[SEP]':
                break
        start=i+1        
        end = len(tokens)
        
        mask = torch.ones((self.seq_len, self.seq_len))
        
        mask[end:,:] = torch.zeros(mask[end:,:].size()) # bottom set to zero for pad
        mask[:,end:] = torch.zeros(mask[:,end:].size()) # right set to zero for pad
        mask[:start, start:end] = torch.zeros(mask[:start, start:end].size()) # set zero for target cannot see source
        mask[start:end, start:end] = torch.tril(mask[start:end, start:end]) # upper triangular diagonal
        
        return mask