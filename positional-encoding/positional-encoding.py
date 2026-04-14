import numpy as np
from math import ceil

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    positions = np.arange(seq_len).reshape(seq_len, 1)
    half_dim = ceil(d_model / 2)

    frequency = np.arange(half_dim).reshape(1, half_dim)
    frequency = 1 / (base ** (2 * frequency / d_model))

    angles = positions @ frequency  # (seq_len, half_dim)

    pe = np.zeros((seq_len, d_model))

    pe[:, 0::2] = np.sin(angles[:, :pe[:, 0::2].shape[1]])
    pe[:, 1::2] = np.cos(angles[:, :pe[:, 1::2].shape[1]])

    return pe