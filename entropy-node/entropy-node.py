import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    values, counts = np.unique(y, return_counts=True)
    probabs = counts / sum(counts)
    log_part = np.log2(probabs)
    return -np.dot(probabs, log_part)
