import numpy as np

def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error.
    """
    n = len(y_true)
    ece = 0.0

    # Store labels and probabilities for each bin
    bins = [[] for _ in range(n_bins)]

    for y, p in zip(y_true, y_pred):
        if p == 1.0:
            idx = n_bins - 1
        else:
            idx = int(p * n_bins)
        bins[idx].append((y, p))

    for b in bins:
        if not b:
            continue

        size = len(b)
        acc = sum(y for y, _ in b) / size
        conf = sum(p for _, p in b) / size
        ece += (size / n) * abs(acc - conf)

    return ece
    