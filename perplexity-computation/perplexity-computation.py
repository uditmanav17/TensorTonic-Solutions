import numpy as np

def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    prob_distributions = np.asarray(prob_distributions)
    actual_tokens = np.asarray(actual_tokens)

    probs = prob_distributions[np.arange(actual_tokens.size), actual_tokens]
    return np.exp(-np.mean(np.log(probs)))
        