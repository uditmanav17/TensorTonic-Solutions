import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    if x.ndim < 3:
        raise ValueError("Input must have at least 3 dimensions")

    axis = tuple(range(x.ndim - 2, x.ndim))
    # print(axis)
    return np.mean(x, axis=axis)
