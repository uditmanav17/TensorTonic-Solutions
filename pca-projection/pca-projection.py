import numpy as np

def pca_projection(X: list, k: int) -> list:
    """
    Returns the centered data projected onto the top components.
    """
    # Convert input to a numpy array for numerical operations
    X_arr = np.array(X, dtype=float)
    
    # 1. Center the data by subtracting the column means
    X_centered = X_arr - np.mean(X_arr, axis=0)
    
    # 2. Compute the covariance matrix using n-1 (sample covariance)
    # np.cov automatically uses ddof=1 (n-1) and expects features as rows if rowvar=True,
    # so we set rowvar=False because columns represent our features.
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # 3. Find the top-k eigenvectors ordered by decreasing eigenvalue
    # eigh returns them in ascending order
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Slice the last k columns (largest eigenvalues) and reverse them for decreasing order
    top_k_eigenvectors = eigenvectors[:, -k:][:, ::-1]
    
    # 4. Project the centered data onto these k eigenvectors
    # Shape: (n_samples, n_features) X (n_features, k) -> (n_samples, k)
    X_projected = np.dot(X_centered, top_k_eigenvectors)
    
    # 5. Return an n x k list of floats
    return X_projected.tolist()
