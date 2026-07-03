import numpy as np
from scipy.special import comb
from scipy import stats

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    # 1. Exact probability of getting exactly k successes (PMF)
    pmf = stats.binom.pmf(k, n, p)

    # 2. Cumulative probability of getting k or fewer successes (CDF)
    cdf = stats.binom.cdf(k, n, p)

    return pmf, cdf

