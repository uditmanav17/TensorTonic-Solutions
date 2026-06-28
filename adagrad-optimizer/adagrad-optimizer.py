import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    G = np.asarray(G)
    g = np.asarray(g)
    w = np.asarray(w)
    
    G_new = G + g * g
    w_new = w - lr * g/ np.sqrt(G_new + eps)
    return w_new, G_new
    