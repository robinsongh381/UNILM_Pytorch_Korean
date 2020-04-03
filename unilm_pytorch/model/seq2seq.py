import torch.nn as nn

from .unilm import UNILM
from .embedding import UNILMEmbedding

class Sequence2Sequence(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, unilm: UNILM, vocab_size, droput, d_model, dec_head, dec_layer):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.embedding = UNILMEmbedding(vocab_size=vocab_size, embed_size=d_model)
        self.unilm = unilm
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=dec_head)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=dec_layer)
        
        self.proj_vocab_layer = nn.Linear(in_features=d_model, out_features=vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.dropout = nn.Dropout(droput)
        
    def forward(self, x, dec_input, segment_label, seq_mask):
        last_layer_hidden_state = self.unilm(x, segment_label, seq_mask)
        last_layer_hidden_state = self.dropout(last_layer_hidden_state)
        
        dec_input_embed = self.embedding(dec_input, segment_label=None)
        decoder_out = self.transformer_decoder(tgt=dec_input_embed,
                                               memory=last_layer_hidden_state)
        logits = self.proj_vocab_layer(decoder_out)
        # softmax = self.softmax(logits)
  
        return logits


    
    
class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
