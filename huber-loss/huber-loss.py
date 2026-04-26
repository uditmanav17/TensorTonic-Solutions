import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    # Write code here
    # Convert inputs to numpy arrays for vectorization
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate the element-wise absolute error
    error = np.abs(y_true - y_pred)
    
    # Apply the piecewise formula
    loss = np.where(
        error <= delta,
        0.5 * (error ** 2), # Quadratic for small errors
        delta * error - 0.5 * (delta ** 2) # Linear for large errors
    )
    
    # Return the average loss across the batch
    return np.mean(loss)