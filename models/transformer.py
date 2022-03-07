import torch
import torch.nn as nn
import numpy as np
import math

class PositionalEncoding(nn.Module):
    # taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model:int, dropout:float, max_len:int):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # the formula from "Attention is all you need"
        pe[:, 0::2] = torch.sin(position * div_term) # for the uneven numbers
        pe[:, 1::2] = torch.cos(position * div_term) # for the even numbers
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x_enc = x + self.pe[:x.size(0), :]
        return self.dropout(x_enc)


class Network(nn.Module):
    def __init__(self, input_size:int, output_size:int, hidden_size:int=512, num_layers:int=5, dropout:float=0.01, no_heads:int=8, maxlen:int=1000):
        super().__init__()
        assert (hidden_size%2 == 0) and (hidden_size%no_heads == 0)
        torch.cuda.empty_cache()
        self.inlayer = nn.Linear(input_size, hidden_size)
        self.posenc = PositionalEncoding(hidden_size, dropout, maxlen)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=no_heads, dim_feedforward=hidden_size, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.outlayer = nn.Linear(hidden_size, output_size)
        
        # move model to GPU:
        if torch.cuda.is_available():
            self.cuda()
        self.device = self.inlayer.weight.device
        
    def forward(self, x):
        # x input is batch, seq, features
        h = self.inlayer(x)
        h = self.posenc(h.permute(1,0,2))
        h = self.encoder(h)[-1]
        y = self.outlayer(h).unsqueeze(0).permute(1,0,2)
        return y


def sample(data, nans:object, targets:list, inputlength:int=100, batchsize:int=50, device:str='cpu', importance:float=0.75):
    """
    importance sampling
    """
    # sample the indexes:
    indexes = np.arange(inputlength, data.shape[0] - 1) # possible indexes to choose from
    indexes = np.array(list(set(indexes).difference(nans)))
    probabilities = np.exp(importance*indexes / indexes.shape[0])
    probabilities /= probabilities.sum()
    i = np.random.choice(indexes, batchsize, p=probabilities, replace=False) # chosen indexes
    # get the sampled dataframe:
    x = torch.tensor([data.iloc[ii:ii+inputlength].to_numpy() for ii in i], dtype=torch.float).to(device)
    y = torch.tensor([data.iloc[ii+inputlength:ii+inputlength+1][targets].to_numpy() for ii in i], dtype=torch.float).to(device)
    return x, y