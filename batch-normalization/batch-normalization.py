import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Write code here
    beta = np.asarray(beta)
    gamma = np.asarray(gamma)
    x = np.asarray(x)
    ndim = x.ndim
    
    # 1. Determine axes to reduce
    if ndim == 4:
        axes = (0, 2, 3)
        view_shape = (1, -1, 1, 1)
    else:
        axes = (0,)
        view_shape = (1, -1)

    # 2. Compute stats
    mean = np.mean(x, axis=axes, keepdims=True)
    var = np.var(x, axis=axes, keepdims=True)
    
    # 3. Normalize
    x_hat = (x - mean) / np.sqrt(var + eps)
    
    # 4. Reshape gamma and beta for broadcasting
    gamma = gamma.reshape(view_shape)
    beta = beta.reshape(view_shape)
    
    return gamma * x_hat + beta
