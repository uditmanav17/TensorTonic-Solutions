import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    arr = np.array(x)
    # print(arr)
    deno = 1 + np.exp(-arr)
    return 1 / deno
    
    