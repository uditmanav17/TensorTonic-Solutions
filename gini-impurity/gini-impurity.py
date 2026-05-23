import numpy as np

def gini_impurity(left, right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    left = np.array(left)
    right = np.array(right)
    
    l_gini = r_gini = 0.0
    
    l_len, r_len = len(left), len(right)
    total = l_len + r_len

    if l_len:
        l_values, l_counts = np.unique(left, return_counts=True)
        l_gini = l_len / total * (1 - np.sum((l_counts / l_len) ** 2))

    if r_len:
        r_values, r_counts = np.unique(right, return_counts=True)
        r_gini = r_len / total * (1 - np.sum((r_counts / r_len) ** 2))

    return l_gini + r_gini
