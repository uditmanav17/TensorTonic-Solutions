def user_based_cf_prediction(similarities: list, ratings: list) -> float:
    """
    Returns the positive-similarity weighted rating prediction.
    """
    weighted_sum = sim_sum = 0
    for sim, rat in zip(similarities, ratings):
        if sim < 0:
            continue
        weighted_sum += sim * rat
        sim_sum += sim
    return weighted_sum / sim_sum if sim_sum != 0 else 0
