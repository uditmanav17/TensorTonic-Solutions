import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    # Write code here
    if len(fpr) != len(tpr) or len(fpr) < 2:
        return None
    auc = np.trapezoid(tpr, x=fpr)
    return auc
