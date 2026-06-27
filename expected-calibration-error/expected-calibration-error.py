import numpy as np

def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_pred)

    # Bin indices
    bins = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)
    # print(bins)

    # Per-bin counts
    counts = np.bincount(bins, minlength=n_bins)
    # print(counts)
    
    # Per-bin sums
    sum_prob = np.bincount(bins, weights=y_prob, minlength=n_bins)
    sum_true = np.bincount(bins, weights=y_true, minlength=n_bins)
    # print(sum_prob)
    # print(sum_true)
    
    # Means (avoid division by zero)
    nonempty = counts > 0
    conf = np.zeros(n_bins)
    acc = np.zeros(n_bins)

    conf[nonempty] = sum_prob[nonempty] / counts[nonempty]
    acc[nonempty] = sum_true[nonempty] / counts[nonempty]

    ece = np.sum((counts[nonempty] / len(y_true)) *
                 np.abs(acc[nonempty] - conf[nonempty]))

    return float(ece)