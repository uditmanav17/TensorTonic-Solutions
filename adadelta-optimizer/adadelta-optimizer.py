import numpy as np

def adadelta_step(
    w: list, 
    grad: list, 
    E_grad_sq: list, 
    E_update_sq: list, 
    rho: float = 0.9, 
    eps: float = 1e-6
) -> dict:
    """
    Returns a dictionary with new_w, new_E_grad_sq, and new_E_update_sq.
    """
    # Write code here
    E_grad_sq = np.asarray(E_grad_sq)
    grad = np.asarray(grad)
    E_update_sq = np.asarray(E_update_sq)
    
    new_E_grad_sq = rho * E_grad_sq + (1 - rho) * grad**2
    
    delta_w = -np.sqrt((E_update_sq + eps) / (new_E_grad_sq + eps)) * grad
    new_E_update_sq = rho * E_update_sq + (1 - rho) * np.square(delta_w)

    new_w = np.asarray(w) + delta_w
    
    return {
        "new_w": new_w, 
        "new_E_grad_sq": new_E_grad_sq, 
        "new_E_update_sq": new_E_update_sq
    }