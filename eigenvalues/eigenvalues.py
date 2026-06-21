import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    # Write code here
    try:
        matrix = np.asarray(matrix)
    except ValueError:
        return None
    mat_dim = matrix.ndim
    mat_shape = matrix.shape
    if mat_dim != 2 or mat_shape[0] != mat_shape[1]:
        return None

    e_vals = np.linalg.eigvals(matrix)
    idx = np.lexsort((e_vals.real, e_vals.imag))
    sorted_eigvals = e_vals[idx]
    return sorted_eigvals
    