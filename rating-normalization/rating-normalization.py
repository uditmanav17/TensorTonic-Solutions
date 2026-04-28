import numpy as np

def rating_normalization(matrix):
    """
    Mean-center each user's ratings in the user-item matrix.
    """
    # Write code here
    arr = np.array(matrix)
    row_means = np.sum(arr, axis=1, keepdims=True) / np.count_nonzero(arr, axis=1, keepdims=True)
    # print(row_means)
    
    return (np.where(arr != 0, arr - row_means, 0)).tolist()
    