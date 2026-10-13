import math
import numpy as np

def label_smoothing_loss(predictions: list, target: int, epsilon: float) -> float:
    """
    Returns cross-entropy loss for the smoothed target distribution.
    """
    preds = np.asarray(predictions, dtype=float)
    # print(preds)
    preds = np.clip(preds, 1e-15, 1.0)
    num_classes = len(preds)

    target_dist = np.full(num_classes, epsilon / num_classes)
    target_dist[target] += 1 - epsilon
    # print(target_dist)

    loss = -np.sum(target_dist * np.log(preds))
    return float(loss)
