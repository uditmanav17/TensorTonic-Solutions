import numpy as np

def item_cf_predict(user_ratings: list, item_similarities: list, target: int) -> float:
    """
    Returns the similarity-weighted rating prediction.
    """
    r = np.asarray(user_ratings, dtype=float)
    s = np.asarray(item_similarities, dtype=float)

    mask = (np.arange(r.size) != target) & (r != 0) & (s > 0)
    s = s[mask]
    r = r[mask]

    return float(np.dot(s, r) / s.sum()) if s.size else 0.0
