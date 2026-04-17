import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    N = len(y_true)
    row_idx = np.arange(N)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    probabs = y_pred[row_idx, y_true]
    L = np.mean(np.log(probabs))
    return -L
