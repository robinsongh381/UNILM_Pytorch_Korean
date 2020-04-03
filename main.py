#!/usr/bin/env python
# coding: utf-8

# In[1]:

import argparse, os
import torch
from torch.utils.data import DataLoader

from unilm_pytorch.model.unilm import UNILM
from unilm_pytorch.trainer.pretrain import UNILMTrainer
from unilm_pytorch.dataset.dataset import UNILMDataset
from unilm_pytorch.tokenizer.kobert_tokenizer import KoBertTokenizer
from unilm_pytorch.tokenizer.tokenizer import Tokenizer

print('Load KoBERT Tokenizer')
kobert_tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
tokenizer = Tokenizer(kobert_tokenizer, add_tokens=True)

# Free GPU cached memory
torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--train_dataset_path", type=str,  default='./_data/articles_body_head.txt', help="train dataset for train bert")
    parser.add_argument("-t", "--test_dataset_path", type=str, default = './data/valid_articles_body_head.txt', help="test set for evaluate train set")
    parser.add_argument("-o", "--output_path", type=str, help="ex)output/bert.model")
    parser.add_argument("-log", "--log_dir", required=False, type=str, default= './log', help="log directory")

    parser.add_argument("-hs", "--hidden", type=int, default=512, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=6, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=256, help="maximum sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker size")
    parser.add_argument("-te", "--test_every", type=int, default=2000, help="test every this step")
    parser.add_argument("-se", "--save_every", type=int, default=20000, help="save every this step")
    parser.add_argument("-dt", "--do_test", type=bool, default=True, help="whether do test during train or not")
    parser.add_argument("-sdt", "--save_during_train", type=bool, default=True, help="whether save during train")
    
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    args = parser.parse_args()
    
    experiment_name = 'enc_{}_maxlen_{}_hidden_{}_heads_{}_batch_{}'.format(args.layers,
                                                                           args.seq_len,
                                                                           args.hidden,
                                                                           args.attn_heads,
                                                                           args.batch_size)
    if not os.path.exists('./experiment/'+experiment_name):
        os.mkdir('./experiment/'+experiment_name)
    log_dir = './experiment/'+experiment_name
    
    # print("Loading Train Dataset from {}".format(args.train_dataset_path))
    train_dataset = UNILMDataset(args.train_dataset_path, tokenizer, seq_len=args.seq_len, corpus_lines=args.corpus_lines, on_memory=args.on_memory)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    
    # print("Loading Test Dataset from {}".format(args.test_dataset_path))
    test_dataset = UNILMDataset(args.test_dataset_path, tokenizer, seq_len=args.seq_len, corpus_lines=args.corpus_lines, on_memory=args.on_memory)
    test_data_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers)

    print("Building UNILM model")
    unilm = UNILM(len(tokenizer), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)
    
    print("Creating UNILM Trainer")
    trainer = UNILMTrainer(unilm, tokenizer, train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                              lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                              with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, log_dir = log_dir, args=args)


    print("Training Start")
    trainer.train(args.epochs)
    
#     for epoch in range(args.epochs):
#         trainer.train(epoch)
#         trainer.save(epoch, './')
        
#         if test_data_loader is not None:
#             trainer.test(epoch)



