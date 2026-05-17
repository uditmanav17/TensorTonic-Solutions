import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    # Convert inputs to numpy arrays
    anchor = np.array(anchor)
    positive = np.array(positive)
    negative = np.array(negative)
    
    # Compute squared Euclidean distances along the last axis
    d_ap = np.sum((anchor - positive) ** 2, axis=-1)
    d_an = np.sum((anchor - negative) ** 2, axis=-1)
    
    # Compute the element-wise loss using np.maximum
    losses = np.maximum(0.0, d_ap - d_an + margin)
    
    # Return the scalar mean loss across the batch
    return float(np.mean(losses))