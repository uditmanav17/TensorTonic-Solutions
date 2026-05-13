import numpy as np
from math import factorial

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Write code here
    cdf = 0
    for i in range(k + 1):
        pmf = np.exp(-lam) * lam ** i / factorial(i)
        cdf += pmf
    return pmf, cdf
