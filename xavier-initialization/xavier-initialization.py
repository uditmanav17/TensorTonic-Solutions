import numpy as np

def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    # Write code here
    W = np.array(W)
    L = (6 / (fan_in + fan_out)) ** 0.5
    return W * 2 * L - L
