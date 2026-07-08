import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    N = len(y_pred)
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    return np.linalg.norm(y_pred - y_true, ord=2) ** 2 / N
