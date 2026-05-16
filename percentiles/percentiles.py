import numpy as np

def percentiles(x, q):
    """ Compute percentiles using linear interpolation (Inclusive method). """
    x = np.array(x, dtype=float)
    return np.quantile(x, [i/100 for i in q])