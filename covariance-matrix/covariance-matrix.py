import numpy as np

def covariance_matrix(X):
    # Convert to numpy array and ensure it's at least 2D
    x = np.asarray(X)
    
    # Requirement: Return None if not 2D or N < 2
    if x.ndim != 2 or x.shape[0] < 2:
        return None
        
    N, D = x.shape
    
    # 1. Center the data (Subtract mean of each feature)
    # axis=0 calculates the mean down the columns
    x_centered = x - np.mean(x, axis=0)
    
    # 2. Compute Covariance Matrix
    # Using the dot product of the transposed centered matrix
    # Shape: (D, N) @ (N, D) -> (D, D)
    cov = (x_centered.T @ x_centered) / (N - 1)
    
    return cov