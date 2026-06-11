import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    # Write code here
    p = np.asarray(p)
    y = np.asarray(y)
    loss = (
        - np.pow(1 - p, gamma) * y * np.log(p) 
        - np.pow(p, gamma) * (1 - y) * np.log(1 - p)
    )
    # print(loss.mean())
    return loss.mean()