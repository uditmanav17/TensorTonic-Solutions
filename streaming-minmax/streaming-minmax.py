import numpy as np
from math import inf

def streaming_minmax_init(D):
    """
    Initialize state dict with min, max arrays of shape (D,).
    """
    state = {
        "min": np.array([inf] * D),
        "max": np.array([-inf] * D),
    }
    return state

def streaming_minmax_update(state, X_batch, eps=1e-8):
    """
    Update state's min/max with X_batch, return normalized batch.
    """
    # Min/max for this batch (per feature)
    batch_min = np.min(X_batch, axis=0)
    batch_max = np.max(X_batch, axis=0)

    # Update running statistics
    state["min"] = np.minimum(state["min"], batch_min)
    state["max"] = np.maximum(state["max"], batch_max)

    # Normalize using updated statistics
    normalized = (X_batch - state["min"]) / (state["max"] - state["min"] + eps)

    return normalized