import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    N, num_features = X.shape
    W = np.zeros((num_features, ))
    b = 0

    for _ in range(steps):
        preds = _sigmoid(X @ W + b)
        # loss = - np.mean(y * np.log(preds) + (1 - y) * log(1 - preds))
        dL_dW = 1/N * X.T @ (preds - y)
        # print(f"dL_dW: {dL_dW}")
        # print(f"preds: {preds}")
        dL_db = 1/N * np.sum(preds - y)
        # print(f"dL_db: {dL_db}")
        W -= lr * dL_dW
        b -= lr * dL_db
    return W, b

    
    
    
    
    pass