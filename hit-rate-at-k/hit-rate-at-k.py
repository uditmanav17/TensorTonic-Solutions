import numpy as np

def hit_rate_at_k(recommendations, ground_truth, k):
    """ Compute the hit rate at K using vectorization. """
    if not recommendations:
        return 0.0
        
    # 1. Convert inputs to NumPy arrays and slice top-K recommendations
    preds = np.array(recommendations)[:, :k]
    targets = np.array(ground_truth)
    
    # 2. Use broadcasting to check matches across rows
    # targets[:, :, None] matches the shape to allow element-wise comparisons
    matches = np.any(preds[:, None, :] == targets[:, :, None], axis=(1, 2))
    
    # 3. Take the mean of the boolean array (True = 1, False = 0)
    return float(np.mean(matches))