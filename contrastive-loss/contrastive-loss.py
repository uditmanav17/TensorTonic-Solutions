import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    # Write code here
    a = np.array(a)
    b = np.array(b)
    y = np.array(y)
    
    # 1. Compute Euclidean distance
    D = np.linalg.norm(a - b, axis=-1)
    
    # 2. Similar pairs (y=1 based on your editor's docstring)
    # If y=1, we want D to be 0
    similar_loss = y * np.square(D)
    
    # 3. Dissimilar pairs (y=0 based on your editor's docstring)
    # If y=0, we want D to be at least 'margin'
    dissimilar_loss = (1 - y) * np.square(np.maximum(0, margin - D))
    
    # Total loss
    loss = similar_loss + dissimilar_loss
    
    if reduction == "mean":
        return np.mean(loss)
    return np.sum(loss)