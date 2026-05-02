import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.asarray(x)
    exp_pos = np.exp(x)
    exp_neg = np.exp(-x)
    return (exp_pos - exp_neg) / (exp_pos + exp_neg)
