import numpy as np

def simple_moving_average(values, window_size):
    """
    Compute the simple moving average of the given values.
    """
    # Write code here
    values = np.array(values)
    test = np.cumsum(values, dtype=float)
    test[window_size:] = test[window_size:] - test[:-window_size]

    return (test[window_size - 1:] / window_size).tolist()