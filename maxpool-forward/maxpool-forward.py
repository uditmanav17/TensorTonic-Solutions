from itertools import product
import numpy as np
import math 

def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    # Write code here
    X = np.array(X)
    in_rows, in_cols = X.shape
    out_rows = math.floor((in_rows - pool_size) / stride) + 1
    out_cols = math.floor((in_cols - pool_size) / stride) + 1
    out = np.zeros(shape=(out_rows, out_cols))
    for i, j in product(range(out_rows), range(out_cols)):
        # print(i, j)
        # print(f"rslice = {i*stride} - {i*stride + pool_size}")
        # print(f"cslice = {j*stride} - {j*stride + pool_size}")
        # print(X[i*stride: i*stride + pool_size, j*stride: j*stride + pool_size])
        out[i][j] = np.max(X[i*stride: i*stride + pool_size, j*stride: j*stride + pool_size])
    # print(out)
    return out.tolist()