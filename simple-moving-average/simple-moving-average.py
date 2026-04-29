
import numpy as np

def simple_moving_average(values, window_size):
    # Create a uniform filter: [1/k, 1/k, ..., 1/k]
    weights = np.ones(window_size) / window_size
    # 'valid' mode only returns points where the window fully overlaps
    return np.convolve(values, weights, mode='valid').tolist()
