import numpy as np

def priority_replay_sample(priorities: list, alpha: float, beta: float) -> list:
    """
    Returns sampling probabilities and normalized importance weights.
    """
    N = len(priorities)
    p = np.asarray(priorities)
    
    p_alpha = p ** alpha
    p_sampling = p_alpha / p_alpha.sum()
    
    sampling_wts = np.pow(N * p_sampling, -beta)
    normalized_wts = sampling_wts / np.max(sampling_wts)
    
    return [p_sampling.tolist(), normalized_wts.tolist()]
