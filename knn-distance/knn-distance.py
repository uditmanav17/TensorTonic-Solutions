import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    # Write code here
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    # Convert 1D -> (n_samples, 1)
    if X_train.ndim == 1:
        X_train = X_train[:, None]

    if X_test.ndim == 1:
        X_test = X_test[:, None]
        
    # (n_test, n_train, n_features)
    diff = X_test[:, None, :] - X_train[None, :, :]

    # (n_test, n_train)
    dists = np.sqrt(np.sum(diff ** 2, axis=2))

    # k nearest neighbors for each test point
    top_k = np.argsort(dists, axis=1)[:, :k]

    top_k = np.pad(top_k, ((0, 0), (0, max(0, k - top_k.shape[1]))), constant_values=-1)
    return top_k
