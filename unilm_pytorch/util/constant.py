
train_dataset_path  = './kor_data/train_articles_body_head.txt'
test_dataset_path = './kor_data/valid_articles_body_head.txt'

vocab = tokenizer
corpus_lines = None
on_memory = False

hidden=256
layers=8
attn_heads=8
seq_len = 128

batch_size=16
epochs=50
num_workers=0

with_cuda=True
log_freq=1e+10
corpus_lines=None
cuda_devices = None
on_memory = True

lr =1e-3
adam_weight_decay = 0.01
adam_beta1 = 0.9
adam_beta2 = 0.999