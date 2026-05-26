import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    # Your code here
    return np.sqrt(np.sum(np.square(v), axis=-1))