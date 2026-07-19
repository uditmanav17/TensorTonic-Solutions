import numpy as np

def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    
    return np.linalg.inv(X.T @ X + lam * np.eye(X.shape[1])) @ X.T @ y
    