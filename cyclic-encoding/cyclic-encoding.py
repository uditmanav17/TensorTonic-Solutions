import numpy as np

def cyclic_encoding(values, period):
    """
    Encode cyclic features as sin/cos pairs.
    """
    # Write code here
    values = np.array(values, dtype=float)
    theta = 2 * np.pi * values / period
    sin = np.sin(theta)
    cos = np.cos(theta)
    return [[sin[i], cos[i]] for i in range(len(sin))]
