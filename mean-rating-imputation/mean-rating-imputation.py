import numpy as np

def mean_rating_imputation(ratings_matrix: list, mode: str) -> list:
    """
    Returns a copy with missing ratings replaced by user or item means.
    """
    ratings = np.asarray(ratings_matrix, dtype=float)

    if mode == "user":
        counts = np.count_nonzero(ratings, axis=1)
        means = np.divide(
            ratings.sum(axis=1),
            counts,
            out=np.zeros(ratings.shape[0]),
            where=counts != 0
        )
        return np.where(ratings == 0, means[:, None], ratings).tolist()

    elif mode == "item":
        counts = np.count_nonzero(ratings, axis=0)
        means = np.divide(
            ratings.sum(axis=0),
            counts,
            out=np.zeros(ratings.shape[1]),
            where=counts != 0
        )
        return np.where(ratings == 0, means[None, :], ratings).tolist()
