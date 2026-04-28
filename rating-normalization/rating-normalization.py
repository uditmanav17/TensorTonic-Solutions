import numpy as np

def rating_normalization(matrix):
    arr = np.array(matrix, dtype=float)
    
    # Replace 0s with NaN so they are ignored in mean calculations
    arr[arr == 0] = np.nan
    
    # Calculate mean along rows, ignoring NaNs
    # Use nan_to_num to handle rows that are all zeros (returns 0 instead of NaN)
    row_means = np.nanmean(arr, axis=1, keepdims=True)
    
    # Subtract means where the original value wasn't 0
    normalized = arr - row_means
    
    # Convert back: NaNs (originally 0s) back to 0.0
    return np.nan_to_num(normalized).tolist()