import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Perform one Nadam update step.
    """
    # Write code here
    m = np.asarray(m)
    w = np.asarray(w)
    v = np.asarray(v)
    grad = np.asarray(grad)
    
    new_m = beta1 * m + (1 - beta1) * grad
    new_v = beta2 * v + (1 - beta2) * np.square(grad)
    new_w = w - lr * (beta1 * new_m + (1 - beta1) * grad) / ( np.sqrt(new_v) + eps)
    
    return new_w, new_m, new_v
