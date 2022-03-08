# krafthack submission Versiro

Contains a time series transformer model with an input sequence length of 4 minutes, a batch size of 50 per training step, 6 layers, 8 heads and a layer size of 96.
The inputs are sampled via importance sampling and a dropout is applied on the targets, but not on the features.
