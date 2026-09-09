import numpy as np

def seasonal_average(series: list, period: int) -> list:
    """
    Returns the average for each position in the seasonal cycle.
    """
    ser = np.asarray(series).reshape((-1, period))
    ans = np.mean(ser, axis=0)
    return ans.tolist()
