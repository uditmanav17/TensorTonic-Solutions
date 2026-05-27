
from scipy import stats as sp_stat

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Write code here
    pmf = sp_stat.poisson.pmf(mu=lam, k=k)
    cdf = sp_stat.poisson.cdf(mu=lam, k=k)
    return pmf, cdf
    