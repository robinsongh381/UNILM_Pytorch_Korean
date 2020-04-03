# UNILM - Pytorch

Pytorch implementation of Microsoft's Unified Language Model Pre-training for Korean

> Unified Language Model Pre-training
> Paper URL : https://arxiv.org/abs/1905.03197


## Acknowledgement

A large proportion of this implementation is from [BERT-pytorch](https://github.com/codertimo/BERT-pytorch)

## Data
Place a train text file and a test file (both are one sentence per line) into `./data` directory and set each path for `main.py`'s `train_dataset_path` and `test_dataset_path arguments`

I have used crawled Korean articles' head for both train and test text files 

## Tokenizer
Since the model is trained for Korean, I have used Korean sentencepiece tokenizer from [KoBERT](https://github.com/SKTBrain/KoBERT)

## Train
```
python main.py
```

## Language Model Pre-training

In the paper, authors shows the new language model training methods, 
which are "masked language model" and "predict next sentence".


### Masked Language Model 

> Original Paper : 3.3.1 Task #1: Masked LM 

```
Input Sequence  : The man went to [MASK] store with [MASK] dog
Target Sequence :                  the                his
```

#### Rules:
Randomly 15% of input token will be changed into something, based on under sub-rules

1. Randomly 80% of tokens, gonna be a `[MASK]` token
2. Randomly 10% of tokens, gonna be a `[RANDOM]` token(another word)
3. Randomly 10% of tokens, will be remain as same. But need to be predicted.

#### Attention Masking
As stated in the paper, within one training batch, 1/3 of the time we use the `bidirectional` LM objective, 1/3 of
the time we employ the `sequence-to-sequence` LM objective, and both `left-to-right` and `right-to-left`
LM objectives are sampled with rate of 1/6.

Please refer to [dataset.py](./unilm_pytorch/dataset/dataset.py)
