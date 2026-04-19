import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.array(x)
    
    # 1. Generate random numbers using the correct method
    if rng is not None:
        random_values = rng.random(x.shape)
    else:
        random_values = np.random.random(x.shape)
    
    # 2. Create mask: Keep if random_value >= p
    # This aligns with p being the "probability of dropping"
    mask = (random_values >= p).astype(float)
    
    # 3. Apply Inverted Dropout scaling
    # Scale by 1/(1-p) to maintain expected value
    scale = 1 / (1 - p)
    output = x * mask * scale
    return output, mask * scale