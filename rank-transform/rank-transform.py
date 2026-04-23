import numpy as np

def rank_transform(values):
    """
    Replace each value with its average rank.
    """
    values = np.array(values)
    
    # 1. Get the indices that would sort the array
    # 2. Get the ranks (0 to n-1) for those sorted positions
    # 3. Use np.unique to find ties and calculate their average positions
    
    # Alternatively, the most direct way in the scientific stack:
    from scipy.stats import rankdata
    return list(rankdata(values, method='average'))

# If you need a pure NumPy implementation without scipy:
def rank_transform_numpy(values):
    values = np.array(values)
    # Get sorted indices and the inverse mapping
    sorter = np.argsort(values)
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(values))
    
    # Find unique values and their counts to handle ties
    unique_vals, counts = np.unique(values, return_counts=True)
    
    if np.all(counts == 1):
        return (inv + 1).astype(float)
        
    # Calculate average ranks for ties
    cumulative_counts = np.cumsum(counts)
    # Average rank formula: (start_rank + end_rank) / 2
    avg_ranks = (cumulative_counts - counts + 1 + cumulative_counts) / 2.0
    
    # Map the average ranks back to the original positions
    val_to_avg_rank = dict(zip(unique_vals, avg_ranks))
    return np.array([val_to_avg_rank[x] for x in values])