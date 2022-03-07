import torch
import torch.nn as nn
import numpy as np

class Network(nn.Module):
    """ 
    LSTM model
    """
    def __init__(self, input_size:int, output_size:int, layer_sizes:list=[15,15,15]):
        super().__init__()
        # save information about layers:
        for l in layer_sizes:
            assert type(l) == int
        self.no_layers = len(layer_sizes) 
        # make the encoder:
        layers = []
        h_size = input_size
        for l in layer_sizes:
            layers.append(nn.LSTMCell(h_size, l))
            h_size = l
        self.encoder = nn.ModuleList(layers)
        # decoder is just a single layer:
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(layer_sizes[-1], output_size)
        # move model to GPU:
        if torch.cuda.is_available():
            self.cuda()
        self.device = self.decoder.weight.device
        
    def forward(self, x:object, hidden:tuple=None):
        # read hidden state, if available:
        if hidden is not None:
            h_prev = [h.detach() for h in hidden[0]]
            c_prev = [c.detach() for c in hidden[1]]
        else:
            h_prev = [None for _ in range(self.no_layers)]
            c_prev = [None for _ in range(self.no_layers)]
        # container for new hidden states:
        h_new = []
        c_new = []
        # iterate through encoder:
        h = h_prev[0]
        c = c_prev[0]
        z = x.detach()
        for i, l in enumerate(self.encoder):
            if c is None:
                h, c = l(z)
            else:
                h, c = l(z, (h, c))
            # update the transition between layers:
            z = h
            # store the hidden states:
            h_new.append(h)
            c_new.append(c)
            # tab through hidden states:
            h = h_prev[i]
            c = c_prev[i]
        # decoder layer:
        y = self.decoder(self.relu(z))
        return y, (h_new, c_new)

def make_tensor(df:object, device:object, columns:list=[None], nancolumn:int=0):
    """
    create a tensor from the given dataframe and puts it on the given device
    """
    if columns[0] is None:
        columns = df.columns
    # find nans:
    nans = df[columns[nancolumn]].isna()
    df = df.fillna(0)
    # nans:
    t = torch.tensor(df[columns].to_numpy(), dtype=torch.float).to(device)
    return t, nans


def slice_df(df:object, maxlen:int=10e3):
    """
    slice the given dataframe into chunks of the maximum length given
    """
    no_slices = int(np.ceil(df.shape[0]/maxlen)) # in how many slices the dataframe is split into
    slices = np.split(df, no_slices)
    return slices
