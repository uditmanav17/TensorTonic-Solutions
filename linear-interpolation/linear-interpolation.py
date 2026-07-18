import numpy as np

def linear_interpolation(values):
    """
    Fill missing (None) values using linear interpolation.
    """
    # Write code here
    arr = np.array(
        [np.nan if v is None else float(v) for v in values],
        dtype=float
    )

    x = np.arange(len(arr))
    mask = ~np.isnan(arr)

    arr[~mask] = np.interp(x[~mask], x[mask], arr[mask])

    return arr.tolist()
