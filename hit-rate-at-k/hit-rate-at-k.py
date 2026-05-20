def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    # Write code here
    hits = 0
    for rec, truth in zip(recommendations, ground_truth):
        hits += 1 if set(rec[:k]).intersection(truth) else 0
    return hits / len(recommendations)
        