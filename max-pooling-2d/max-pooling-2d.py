import numpy as np

def max_pooling_2d(X: list, pool_size: int) -> list:
    """
    Returns non-overlapping maximum-pooled windows.
    """
    X = np.asarray(X)

    h, w = X.shape
    h_out = h // pool_size
    w_out = w // pool_size

    X = X[:h_out * pool_size, :w_out * pool_size]

    X = X.reshape(
        h_out, pool_size,
        w_out, pool_size
    )

    return X.max(axis=(1, 3)).tolist()