def linear_layer_forward(X: list, W: list, b: list) -> list:
    """
    Returns the affine transformation for every input row.
    """
    W = np.asarray(W)
    b = np.asarray(b)
    X = np.asarray(X)
    ans = X @ W + b
    return ans.tolist()
