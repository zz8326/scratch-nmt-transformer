import torch
from torch import nn
from torch.nn import functional as F
import pdb
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq, d_model):
        super(PositionalEmbedding, self).__init__()
        # position index [seq_len, 1]
        position = torch.arange(0, max_seq).unsqueeze(1)  

       
        # 計算編碼值
        div_terms = torch.exp(torch.arange(0, d_model, 2) *  (-math.log(10000.0) / d_model))
        
        # 構造編碼圖
        pe = torch.zeros(max_seq, d_model)
        pe[:, 0::2] = torch.sin(position * div_terms)
        pe[:, 1::2] = torch.cos(position * div_terms)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape = [b, seq, d_model]
        """
        return x + self.pe[:, :x.size(1),:]
    

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % heads == 0

        self.heads = heads
        self.d_model = d_model
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)

        self.fc = nn.Linear(d_model, d_model, bias=False)

        self.scale = (d_model // heads) ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.d_k = d_model // heads

    def forward(self, q_inputs, k_inputs, v_inputs, mask= None):
        """
        x: [b, seq, dim]
        """
        batchsz = q_inputs.size(0)
        # [b, heads, seq_len, d_model//heads]
        q = self.q(q_inputs).view(batchsz,q_inputs.size(1),self.heads, self.d_k).permute(0, 2, 1, 3)
        k = self.k(k_inputs).view(batchsz,k_inputs.size(1),self.heads, self.d_k).permute(0, 2, 1, 3)
        v = self.v(v_inputs).view(batchsz,v_inputs.size(1),self.heads, self.d_k).permute(0, 2, 1, 3)

        # Q@K^T
        att_out = torch.matmul(q, k.transpose(-2, -1))
        att_out = att_out * self.scale

        if mask is not None:
            att_out = att_out.masked_fill(mask, -1e9)
            
        
        scores = F.softmax(att_out, dim = -1)
        # score:[b, heads, seq_q, seq_k] @ V : [b, heads, seq_v, d_model//heads] = [b, heads, seq_q, d_model//heads]
        feature_map = torch.matmul(scores, v).transpose(1, 2).contiguous().reshape(batchsz, -1, self.d_model)
        
        # 合併
        out = self.fc(feature_map)
        out = self.dropout(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, d_model, ffn, dropout):
        super(FeedForward, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, ffn, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ffn, d_model, bias=False),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.fc(x)
    
    
class Encoder_Layer(nn.Module):
    def __init__(self, heads, d_model, ffn, dropout = 0.1):
        super(Encoder_Layer,self).__init__()
        self.att = MultiHeadAttention(heads, d_model, dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.fc = FeedForward(d_model, ffn, dropout=dropout)

    def forward(self, x, enc_padding_mask):
        att_out = self.att(x, x, x, enc_padding_mask)
        att_out = self.norm[0](att_out + x)

        fc_out = self.fc(att_out)
        out = self.norm[1](fc_out + att_out)
        return out
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len, heads, d_model, ffn, num_layers,dropout = 0.1):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEmbedding(max_seq=max_len, d_model=d_model)
        self.enc_layers = nn.ModuleList([Encoder_Layer(heads, d_model, ffn=ffn, dropout=dropout) 
                                         for _ in range(num_layers)])
    def forward(self, src, src_mask):
        """
        src : [batch, seq]
        """

        x = self.embed(src)
        x = self.pos(x)
        for layer in self.enc_layers:
            x  = layer(x, src_mask)

        return x
    

class Decoder_Layer(nn.Module):
    def __init__(self, heads, d_model, ffn, dropout = 0.1):
        super(Decoder_Layer, self).__init__()
        self.att = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.fc = FeedForward(d_model, ffn, dropout=dropout)

    def forward(self, dst_x, encoder_kv, enc_padding_mask, tgt_mask):
        # Masked MultiHead Attention
        masked_att_out = self.att(dst_x, dst_x, dst_x, tgt_mask)
        masked_att_out = self.norm[0](dst_x + masked_att_out)

        # Cross Attention
        cross_att_out = self.att(masked_att_out, encoder_kv, encoder_kv, enc_padding_mask)
        cross_att_out = self.norm[1](cross_att_out + masked_att_out)

        out = self.fc(cross_att_out)
        out = self.norm[2](out + cross_att_out)

        return out
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len, heads, ffn, d_model, num_layers, dropout = 0.1):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEmbedding(max_len, d_model)
        self.layers = nn.ModuleList([Decoder_Layer(heads, d_model, ffn=ffn, dropout=dropout) 
                                         for _ in range(num_layers)])
    def forward(self, dst, encoder_kv, enc_padding_mask, tgt_mask):
        x = self.embed(dst)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, encoder_kv, enc_padding_mask,tgt_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, max_len, heads, d_model, ffn, num_layers, padding_idx = 1, dropout = 0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(vocab_size, max_len, heads, d_model, ffn, num_layers, dropout = dropout)
        self.decoder = Decoder(vocab_size, max_len, heads, ffn, d_model, num_layers, dropout=dropout)
        self.fc = nn.Linear(d_model, vocab_size)
        self.padding_idx = padding_idx

    def generate_mask(self, src, tgt):

        # padding mask
        src_mask = (src == self.padding_idx).unsqueeze(1).unsqueeze(2)
        tgt_padding_mask = (tgt == self.padding_idx).unsqueeze(1).unsqueeze(2)

        # look Ahead mask
        tgt_look_ahead_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1), device=tgt.device), diagonal=1).unsqueeze(0).unsqueeze(1).bool()
        tgt_mask = tgt_padding_mask | tgt_look_ahead_mask
        return src_mask, tgt_mask
    
    def forward(self,src, dst):
        src_mask, tgt_mask = self.generate_mask(src, dst)
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(dst, enc_out,src_mask, tgt_mask)
        out = self.fc(dec_out)

        return out
if __name__ == "__main__":

    # max_seq = 50
    # d_model = 10


    # pos_embed = PositionalEmbedding(max_seq, d_model)
    
    vocab_size = 100
    max_len = 10
    heads = 2
    d_model = 32
    ffn = 64
    dropout = 0.1
    srcc_seq = 6
    dstt_seq = 3

    model = Transformer(vocab_size=vocab_size, max_len=max_len, 
                        heads=heads, d_model=d_model, ffn=ffn, num_layers=4, dropout=dropout)

    # 建立簡單測試輸入
    batch_size = 2
    src_seq = torch.randint(1, vocab_size, (batch_size, max_len))  # src: [2, 10]
    dst_seq = torch.randint(1, vocab_size, (batch_size, max_len))  # dst: [2, 10]

    # Forward pass
    out = model(src_seq, dst_seq)

    print("Output shape:", out.shape)