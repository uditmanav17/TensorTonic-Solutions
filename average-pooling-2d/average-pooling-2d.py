import numpy as np
from itertools import product

def average_pooling_2d(X, pool_size):
    """
    Apply 2D average pooling with non-overlapping windows.
    """
    # Write code here
    X = np.asarray(X)
    ROWS, COLS = X.shape
    out_h, out_w = ROWS // pool_size, COLS // pool_size
    pool_out = np.zeros((out_h, out_w))
    for i, j in product(range(out_h), range(out_w)):
        x_start = i * pool_size
        x_end = (i + 1) * pool_size
        y_start = j * pool_size
        y_end = (j + 1) * pool_size
        pool_out[i, j] = X[x_start:x_end , y_start:y_end].mean()
    return pool_out.tolist()
        