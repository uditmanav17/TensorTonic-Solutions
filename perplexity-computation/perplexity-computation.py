import numpy as np

def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    H = 0
    for idx, tkn in enumerate(actual_tokens):
        H += np.log(prob_distributions[idx][tkn])
    H = -H / len(actual_tokens)
    return np.exp(H)
        