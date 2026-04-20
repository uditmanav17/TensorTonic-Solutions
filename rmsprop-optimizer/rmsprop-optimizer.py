import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    w = np.asarray(w, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)

    s_new = beta * s + (1 - beta) * g ** 2
    w_new = w - lr * g / (s_new + eps) ** 0.5

    return w_new, s_new
    

    