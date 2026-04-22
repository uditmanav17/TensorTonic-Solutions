import numpy as np



def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here
    dims = x.ndim
    print(dims)
    if dims == 3:
        return np.mean(x, axis=(1, 2))
    if dims == 4:
        return np.mean(x, axis=(2, 3))
    else:
        raise ValueError("Mismatched dimensions!")