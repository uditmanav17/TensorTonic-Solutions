import numpy as np

def percentiles(x, q):
    """ Compute percentiles using linear interpolation (Inclusive method). """
    x = np.array(x, dtype=float)
    x.sort()
    N = len(x)
    ans = []
    
    for quantile in q:
        # Calculate the precise floating-point index using the inclusive method (N - 1)
        virtual_idx = (quantile / 100) * (N - 1)
        
        # Separate into the floor index and the fractional remainder
        idx = int(np.floor(virtual_idx))
        rem = virtual_idx - idx
        
        # Handle the upper boundary neatly
        if idx >= N - 1:
            curr_q = x[-1]
        else:
            # Linear interpolation formula
            curr_q = x[idx] + rem * (x[idx + 1] - x[idx])
            
        ans.append(curr_q)
        
    return np.array(ans)