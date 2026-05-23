import numpy as np

def gini_impurity(left, right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    left = np.asarray(left)
    right = np.asarray(right)
    
    n_l, n_r = left.size, right.size
    total = n_l + n_r
    
    if total == 0:
        return 0.0
    
    gini_split = 0.0
    
    for child, size in [(left, n_l), (right, n_r)]:
        if size > 0:
            _, counts = np.unique(child, return_counts=True)
            # 1 - sum(p_i^2)
            gini_node = 1.0 - np.sum((counts / size) ** 2)
            # Accumulate weighted impurity
            gini_split += (size / total) * gini_node
            
    return gini_split