import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code here
    g = np.array(g)
    g_norm = np.linalg.norm(g)
    if g_norm <= max_norm or max_norm <= 0:
        return g
    return g * max_norm / g_norm