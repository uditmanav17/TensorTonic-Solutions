import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    # Simple error handling for invalid inputs
    try:
        matrix = np.array(matrix, dtype=float)
    except (ValueError, TypeError):
        return None
        
    if matrix.ndim != 2:
        return None

    # Check for invalid axis values to prevent out-of-bounds exceptions
    if axis not in [0, 1, -1, -2, None]:
        return None

    # Step 1: Calculate the absolute values for L1 and Max norms
    abs_matrix = np.abs(matrix)

    # Step 2: Compute the norm based on the type
    match norm_type:
        case "l1":
            norm = np.sum(abs_matrix, axis=axis, keepdims=True)
        case "l2":
            norm = np.sqrt(np.sum(matrix ** 2, axis=axis, keepdims=True))
        case "max":
            norm = np.max(abs_matrix, axis=axis, keepdims=True)
        case _:
            return None

    # Step 3: Handle division by zero safely
    # If the norm is 0, replace it with 1 so dividing leaves it as 0
    norm = np.where(norm == 0, 1, norm)

    return matrix / norm