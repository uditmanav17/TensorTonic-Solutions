import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    # YOUR CODE HERE
    h_list = []
    _, T, _ = X.shape
    for t in range(T):
        x = X[:, t, :]
        h_0 = np.tanh(x @ W_xh.T + h_0 @ W_hh.T + b_h)
        h_list.append(h_0)
    return np.stack(h_list, axis=1), h_list[-1]
