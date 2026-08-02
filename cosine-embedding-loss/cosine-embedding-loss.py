import numpy as np

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    cos_sim = x1 @ x2 / (np.linalg.norm(x1) * np.linalg.norm(x2))

    if label == 1:
        loss = 1 - cos_sim
    else:
        loss = max(0, cos_sim - margin)

    return loss
    