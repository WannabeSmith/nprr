"""
Differentially private confidence intervals and sequences via NPRR and the Laplace mechanisms
"""
from functools import reduce
import warnings
from operator import add
import math
from typing import Callable, Tuple, NamedTuple, Union
from confseq.betting_strategies import lambda_predmix_eb
import numpy as np
from scipy.stats import binom
from confseq.boundaries import beta_binomial_log_mixture

from nprr.cgf import psi_E, psi_H, psi_B, psi_L

from confseq.betting import (
    cs_from_martingale,
)

from confseq.conjmix_bounded import rho2_opt

from nprr.types import RealArray, IntArray


class PrivacyMechanism(NamedTuple):
    """
    Privacy mechanism

    Attributes
    ----------
    name, String
        Name of the privacy mechanism
    """

    name: str
    eps: float
    mechanism_fn: Callable[[RealArray], RealArray]


def prpl_hoeffding_lambda(
    t: IntArray,
    alpha: float = 0.1,
    trunc: float = 1,
    fixed_n: Union[None, int] = None,
) -> RealArray:
    """
    Computes a default value of lambda for the predictable plug-in Hoeffding bound

    Parameters
    ----------
    t, ArrayLike
        Time points
    alpha, float
        desired type-I error level
    trunc, float
        truncation parameter
    fixed_n, int
        fixed sample size if deriving a fixed-n bound

    Returns
    -------
    lambdas, ArrayLike
        lambda values
    """
    if fixed_n is not None:
        rate = np.repeat(fixed_n, t[-1])
    else:
        rate = t * np.log(t + 1)

    return np.minimum(trunc, np.sqrt(8 * np.log(1 / alpha) / rate))


def prpl_laplace_hoeffding_lambda(
    t: IntArray,
    scale: float,
    trunc_scale: float = 0.1,
    alpha: float = 0.1,
    fixed_n: Union[None, int] = None,
) -> RealArray:
    """
    Computes the lambda for the predictable plug-in Laplace-Hoeffding bound

    Parameters
    ----------
    t, ArrayLike
        Time points
    scale, float
        scale parameter of the Laplace distribution
    trunc_scale, float
        truncation parameter
    alpha, float
        type-I error
    fixed_n, int
        fixed sample size or None
    """
    if fixed_n is not None:
        rate = np.repeat(fixed_n, t[-1])
    else:
        rate = t * np.log(t + 1)

    lambdas = np.minimum(
        trunc_scale / scale,
        np.sqrt(np.log(1 / alpha) / ((1 / 8 + np.power(scale, 2)) * rate)),
    )

    return lambdas


def additive_private_chernoff_cs(
    z: RealArray,
    psi_bounded_fn: Callable[[RealArray], RealArray],
    psi_noise_fn: Callable[[RealArray], RealArray],
    l: RealArray,
    alpha: float = 0.1,
) -> Tuple[RealArray, RealArray]:
    """
    Computes the Chernoff-based confidence sequence for additive mechanisms

    Parameters
    ----------
    z, ArrayLike
        Observed (privatized) data
    psi_bounded_fn, Callable
        Function that computes the CGF upper-bound of the bounded random variables
    psi_noise_fn, Callable
        Function that computes the CGF of the noise
    l, ArrayLike
        lambda values
    alpha, float
        Desired type-I error level

    Returns
    -------
    lower, ArrayLike
        Lower confidence sequence
    upper, ArrayLike
        Upper confidence sequence
    """
    margin = (
        np.log(2 / alpha) + np.cumsum(psi_bounded_fn(l) + psi_noise_fn(l))
    ) / (np.cumsum(l))
    weighted_sample_mean = np.cumsum(l * z) / np.cumsum(l)

    return weighted_sample_mean - margin, weighted_sample_mean + margin


def additive_private_hoeffding_cs(
    z: RealArray,
    psi_noise_fn: Callable[[RealArray], RealArray],
    l: RealArray,
    alpha=0.1,
) -> Tuple[RealArray, RealArray]:
    """
    Computes the Hoeffding-based confidence sequence for additive mechanisms

    Parameters
    ----------
    z, ArrayLike
        Observed (privatized) data
    psi_noise_fn, Callable
        Function that computes the CGF of the noise
    l, ArrayLike
        lambda values
    alpha, float
        Desired type-I error level

    Returns
    -------
    lower, ArrayLike
        Lower confidence sequence
    upper, ArrayLike
        Upper confidence sequence
    """
    lower, upper = additive_private_chernoff_cs(
        z, psi_bounded_fn=psi_H, psi_noise_fn=psi_noise_fn, l=l, alpha=alpha
    )

    return np.maximum(lower, 0), np.minimum(upper, 1)


def laplace_hoeffding_cs(
    z: RealArray,
    scale,
    trunc_scale=0.1,
    lambdas: Union[None, RealArray] = None,
    alpha: float = 0.1,
) -> Tuple[RealArray, RealArray]:
    """
    Computes the Laplace-Hoeffding confidence sequence

    Parameters
    ----------
    z, ArrayLike
        Observed (privatized) data
    scale, float
        Scale parameter of the Laplace distribution
    trunc_scale, float
        Truncation parameter
    lambdas, ArrayLike
        Lambda values
    alpha, float
        Desired type-I error level

    Returns
    -------
    lower, ArrayLike
        Lower confidence sequence
    upper, ArrayLike
        Upper confidence sequence
    """
    t = np.arange(1, len(z) + 1)
    if lambdas is None:
        lambdas = prpl_laplace_hoeffding_lambda(
            t=t, scale=scale, trunc_scale=trunc_scale, alpha=alpha / 2
        )

    mu_hat = np.cumsum(lambdas * z) / np.cumsum(lambdas)

    boundary = (
        np.log(2 / alpha)
        + np.cumsum(np.power(lambdas, 2) / 8 + psi_L(lambdas, scale=scale))
    ) / np.cumsum(lambdas)

    l, u = np.maximum(mu_hat - boundary, 0), np.minimum(1, mu_hat + boundary)

    return l, u


def laplace_hoeffding_ci(
    z: RealArray,
    scale,
    trunc_scale=0.1,
    alpha: float = 0.1,
) -> Tuple[float, float]:
    """
    Computes the Laplace-Hoeffding confidence interval

    Parameters
    ----------
    z, ArrayLike
        Observed (privatized) data
    scale, float
        Scale parameter of the Laplace distribution
    trunc_scale, float
        Truncation parameter
    alpha, float
        Desired type-I error level

    Returns
    -------
    lower, ArrayLike
        Lower confidence interval
    upper, ArrayLike
        Upper confidence interval
    """
    t = np.arange(1, len(z) + 1)
    lambdas = prpl_laplace_hoeffding_lambda(
        t=t,
        scale=scale,
        trunc_scale=trunc_scale,
        alpha=alpha / 2,
        fixed_n=len(z),
    )

    l, u = laplace_hoeffding_cs(
        z=z, scale=scale, trunc_scale=trunc_scale, lambdas=lambdas, alpha=alpha
    )
    l = np.maximum.accumulate(l)
    u = np.minimum.accumulate(u)

    return l[-1], u[-1]


def additive_private_bernoulli_supermartingale(
    z: RealArray,
    m: float,
    psi_noise_fn: Callable[[RealArray], RealArray],
    l: RealArray,
) -> RealArray:
    """
    Computes the supermartingale for sub-Bernoulli random variables under additive mechanisms

    Parameters
    ----------
    z, ArrayLike
        Observed (privatized) data
    m, float
        Mean of the Bernoulli random variables to test against
    psi_noise_fn, Callable
        Function that computes the CGF of the noise
    l, ArrayLike
        lambda values

    Returns
    -------
    martingale, ArrayLike
        Supermartingale
    """
    return np.exp(l * z - psi_B(l, p=m) - psi_noise_fn(l))


def additive_private_bernoulli_cs(
    z: RealArray,
    psi_noise_fn: Callable[[RealArray], RealArray],
    l: RealArray,
    alpha=0.1,
    breaks=1000,
) -> Tuple[RealArray, RealArray]:
    """
    Computes confidence sequence for means of sub-Bernoulli random variables under additive mechanisms

    Parameters
    ----------
    z, ArrayLike
        Observed (privatized) data
    psi_noise_fn, Callable
        Function that computes the CGF of the noise
    l, ArrayLike
        lambda values
    alpha, float
        Desired type-I error level
    breaks, int
        Number of breaks to use for the confidence sequence

    Returns
    -------
    lower, ArrayLike
        Lower confidence sequence
    upper, ArrayLike
        Upper confidence sequence
    """
    mart_fn = lambda z, m: additive_private_bernoulli_supermartingale(
        z, m, psi_noise_fn=psi_noise_fn, l=l
    )
    return cs_from_martingale(z, mart_fn, breaks=breaks, alpha=alpha)


def nprr_transform(a: RealArray, r: float) -> RealArray:
    """
    Computes a debiased random variable that has been privatized via NPRR

    Parameters
    ----------
    a, ArrayLike
        Privatized observation
    r, float
        Probability that NPRR outputted the stochastically rounded rv

    Returns
    -------
    z, ArrayLike
        Debiased privatized observation
    """
    return (a - (1 - r) / 2) / r


def mu_hat_t(
    z: RealArray,
    r: Union[float, RealArray],
    lambdas: Union[None, RealArray] = None,
) -> RealArray:
    """
    Computes the debiased estimate of the mean of privatized random variables

    Parameters
    ----------
    z, ArrayLike
        Privatized observations
    r, float
        Probability that NPRR outputted the stochastically rounded rv
    lambdas, ArrayLike
        Lambda values

    Returns
    -------
    mu_hat, ArrayLike
        Debiased estimate of the mean
    """
    if lambdas is None:
        lambdas = np.array(1)
    if isinstance(r, float):
        r = np.repeat(r, len(z))

    return np.cumsum(lambdas * (z - (1 - r) / 2)) / np.cumsum(r * lambdas)


def nprr_hoeffding_cs(
    z: RealArray,
    r: Union[float, RealArray],
    alpha: float = 0.1,
    trunc: float = 1,
    lambdas: Union[None, RealArray] = None,
) -> Tuple[RealArray, RealArray]:
    """
    Computes the Hoeffding confidence sequence for the mean from NPRR-privatized random variables

    Parameters
    ----------
    z, ArrayLike
        Privatized observations
    r, float
        Probability that NPRR outputted the stochastically rounded rv
    alpha, float
        Desired type-I error level
    trunc, float
        Truncation parameter
    lambdas, ArrayLike
        Lambda values

    Returns
    -------
    lower, ArrayLike
        Lower confidence sequence
    upper, ArrayLike
        Upper confidence sequence
    """
    t = np.arange(1, len(z) + 1)

    if lambdas is None:
        lambdas = prpl_hoeffding_lambda(t, alpha=alpha / 2, trunc=trunc)

    mu_hat = mu_hat_t(z=z, r=r, lambdas=lambdas)

    boundary = (
        np.log(2 / alpha) + np.cumsum(np.power(lambdas, 2) / 8)
    ) / np.cumsum(r * lambdas)

    l, u = np.maximum(0, mu_hat - boundary), np.minimum(1, mu_hat + boundary)

    return l, u


def nprr_twosided_runningmean_cs(
    z: RealArray,
    r: float,
    t_opt: int,
    alpha: float = 0.1,
) -> Tuple[RealArray, RealArray]:
    """
    Computes the two-sided confidence sequence for the running mean given NPRR-privatized random variables

    Parameters
    ----------
    z, ArrayLike
        Privatized observations
    r, float
        Probability that NPRR outputted the stochastically rounded rv
    t_opt, int
        Time to optimize the bound for
    alpha, float
        Desired type-I error level

    Returns
    -------
    lower, ArrayLike
        Lower confidence sequence
    upper, ArrayLike
        Upper confidence sequence
    """
    beta = np.sqrt(rho2_opt(t_opt=t_opt, alpha_opt=alpha))

    t = np.arange(1, len(z) + 1)
    mu_hat = mu_hat_t(z, r)
    boundary = np.sqrt(
        (t * beta**2 + 1)
        / (2 * np.power(t * r * beta, 2))
        * np.log(np.sqrt(t * beta**2) / alpha)
    )

    l, u = mu_hat - boundary, mu_hat + boundary

    return np.maximum(l, 0), np.minimum(u, 1)


def nprr_onesided_runningmean_cs(
    z: RealArray,
    r: float,
    t_opt: int,
    alpha: float = 0.1,
) -> RealArray:
    """
    Computes the one-sided confidence sequence for the running mean given NPRR-privatized random variables

    Parameters
    ----------
    z, ArrayLike
        Privatized observations
    r, float
        Probability that NPRR outputted the stochastically rounded rv
    t_opt, int
        Time to optimize the bound for
    alpha, float
        Desired type-I error level

    Returns
    -------
    lower, ArrayLike
        Lower confidence sequence
    """
    beta = np.sqrt(rho2_opt(t_opt=t_opt, alpha_opt=2 * alpha))

    t = np.arange(1, len(z) + 1)
    mu_hat = mu_hat_t(z, r)
    boundary = np.sqrt(
        (t * beta**2 + 1)
        / (2 * np.power(t * r * beta, 2))
        * np.log(1 + np.sqrt(t * beta**2) / (2 * alpha))
    )

    l = mu_hat - boundary

    return np.maximum(l, 0)


def ipw(obs: RealArray, treatment: IntArray, pi: float):
    """
    Computes the inverse probability weighted pseudo-outcomes (influence functions)

    Parameters
    ----------
    obs, ArrayLike
        Private observations
    treatment, ArrayLike
        Treatment assignment
    pi, float
        Propensity score (probability of treatment)

    Returns
    -------
    ipw_obs, ArrayLike
        Inverse probability weighted pseudo-outcomes
    """
    return obs * treatment / pi - obs * (1 - treatment) / (1 - pi)


def ipw_to_unit_interval(ipw_obs, pi):
    """
    Maps inverse probability weighted pseudo-outcomes to the unit interval

    Parameters
    ----------
    ipw_obs, ArrayLike
        Inverse probability weighted pseudo-outcomes
    pi, float
        Propensity score (probability of treatment)

    Returns
    -------
    phi, ArrayLike
        Mapped pseudo-outcomes
    """
    return (ipw_obs + 1 / (1 - pi)) / (1 / pi + 1 / (1 - pi))


def unit_interval_to_ipw(u, pi):
    """
    Maps numbers from the unit interval to the original scale of the inverse probability weighted pseudo-outcomes (performs the inverse of ipw_to_unit_interval)

    Parameters
    ----------
    phi, ArrayLike
        Mapped pseudo-outcomes
    pi, float
        Propensity score (probability of treatment)

    Returns
    -------
    ipw_obs, ArrayLike
        Result of the inverse mapping
    """
    return -1 / (1 - pi) + (1 / pi + 1 / (1 - pi)) * u


def nprr_abtest_lower_cs(
    pseudo_obs: RealArray,
    r: float,
    t_opt: int,
    pi: float,
    alpha: float = 0.1,
) -> RealArray:
    """
    Computes the lower confidence sequence for the A/B test given NPRR-privatized random variables

    Parameters
    ----------
    pseudo_obs, ArrayLike
        Privatized observations
    r, float
        Probability that NPRR outputted the stochastically rounded rv
    t_opt, int
        Time to optimize the bound for
    pi, float
        Propensity score (probability of treatment)
    alpha, float
        Desired type-I error level

    Returns
    -------
    lower, ArrayLike
        Lower confidence sequence
    """
    lower = nprr_onesided_runningmean_cs(
        pseudo_obs, r=r, t_opt=t_opt, alpha=alpha
    )

    return unit_interval_to_ipw(u=lower, pi=pi)


def nprr_ab_twosided_cs(
    pseudo_obs: RealArray,
    r: float,
    t_opt: int,
    pi: float,
    alpha: float = 0.1,
) -> Tuple[RealArray, RealArray]:
    """
    Computes the two-sided confidence sequence for the A/B test given NPRR-privatized random variables

    Parameters
    ----------
    pseudo_obs, ArrayLike
        Privatized observations
    r, float
        Probability that NPRR outputted the stochastically rounded rv
    t_opt, int
        Time to optimize the bound for
    pi, float
        Propensity score (probability of treatment)
    alpha, float
        Desired type-I error level

    Returns
    -------
    lower, ArrayLike
        Lower confidence sequence
    upper, ArrayLike
        Upper confidence sequence
    """
    lower, upper = nprr_twosided_runningmean_cs(
        pseudo_obs, r=r, t_opt=t_opt, alpha=alpha
    )

    return unit_interval_to_ipw(u=lower, pi=pi), unit_interval_to_ipw(
        u=upper, pi=pi
    )


def nprr_hoeffding_ci(
    z: RealArray,
    r: float,
    alpha: float = 0.1,
    running_intersection=True,
) -> Tuple[float, float]:
    """
    Computes the Hoeffding confidence interval for the mean given NPRR-privatized random variables

    Parameters
    ----------
    z, ArrayLike
        Privatized observations
    r, float
        Probability that NPRR outputted the stochastically rounded rv
    alpha, float
        Desired type-I error level
    running_intersection, bool
        Whether to compute confidence intervals via a running intersection of CSs

    Returns
    -------
    lower, float
        Lower confidence interval
    upper, float
        Upper confidence interval
    """
    t = np.arange(1, len(z) + 1)
    n = t[-1]
    lambdas = prpl_hoeffding_lambda(
        t, alpha=alpha / 2, trunc=math.inf, fixed_n=n
    )

    l, u = nprr_hoeffding_cs(z=z, r=r, alpha=alpha, lambdas=lambdas)

    if running_intersection:
        l = np.maximum.accumulate(l)
        u = np.minimum.accumulate(u)

    return l[-1], u[-1]


def nprr_empbern_cs(
    z: RealArray,
    r: Union[float, RealArray],
    alpha: float = 0.1,
    lambdas: Union[None, RealArray] = None,
) -> Tuple[RealArray, RealArray]:
    """
    Computes the empirical Bernstein confidence sequence for the mean given NPRR-privatized random variables

    Parameters
    ----------
    z, ArrayLike
        Privatized observations
    r, float
        Probability that NPRR outputted the stochastically rounded rv
    alpha, float
        Desired type-I error level
    lambdas, ArrayLike
        Sequence of lambdas to use in the confidence sequence

    Returns
    -------
    lower, ArrayLike
        Lower confidence sequence
    upper, ArrayLike
        Upper confidence sequence
    """
    if isinstance(r, float):
        r = np.repeat(r, len(z))

    # Set lambdas in a sensible default way if None
    lambdas = (
        lambda_predmix_eb(z, alpha=alpha / 2, truncation=0.5)
        if lambdas is None
        else lambdas
    )

    t = np.arange(1, len(z) + 1)
    mu_hat = mu_hat_t(z=z, r=r, lambdas=lambdas)

    zeta_hat = np.cumsum(np.append(1 / 2, z[0 : (len(z) - 1)])) / (t + 1)

    boundary = (
        np.log(2 / alpha)
        + 4 * np.cumsum(np.power(z - zeta_hat, 2) * psi_E(lambdas))
    ) / np.cumsum(r * lambdas)

    return np.maximum(0, mu_hat - boundary), np.minimum(1, mu_hat + boundary)


def nprr_empbern_ci(
    z: RealArray,
    r: Union[float, RealArray],
    alpha: float = 0.1,
    truncation=0.5,
    prior_mean=0.5,
    prior_variance=1 / 8,
):
    """
    Computes the empirical Bernstein confidence interval for the mean given NPRR-privatized random variables

    Parameters
    ----------
    z, ArrayLike
        Privatized observations
    r, float
        Probability that NPRR outputted the stochastically rounded rv
    alpha, float
        Desired type-I error level
    truncation, float
        Truncation parameter for lambdas
    prior_mean, float
        Prior guess of the mean
    prior_variance, float
        Prior guess of the variance

    Returns
    -------
    lower, float
        Lower confidence interval
    upper, float
        Upper confidence interval
    """
    lambdas = lambda_predmix_eb(
        z,
        truncation=truncation,
        fixed_n=len(z),
        prior_mean=prior_mean,
        prior_variance=prior_variance,
    )
    l, u = nprr_empbern_cs(z, r, alpha, lambdas=lambdas)
    l = np.maximum.accumulate(l)
    u = np.minimum.accumulate(u)

    return l[-1], u[-1]


def nprr_hedged_mart(
    z: RealArray,
    mu: float,
    r: float,
    alpha_opt=0.1,
    trunc_scale: float = 4 / 5,
    theta: float = 1 / 2,
    prior_mean=1 / 2,
    prior_variance=1 / 8,
) -> RealArray:
    """
    Computes the hedged martingale for a given mean mu given NPRR-privatized random variables

    Parameters
    ----------
    z, ArrayLike
        Privatized observations
    mu, float
        Hypothesized mean
    r, float
        Probability that NPRR outputted the stochastically rounded rv
    alpha_opt, float
        Type-I error level to optimize the bound for
    trunc_scale, float
        Multiplicative truncation parameter for lambdas
    theta, float
        Hedging factor --- how much to weight positive bets vs negative bets (default 1/2)
    prior_mean, float
        Prior guess for the mean
    prior_variance, float
        Prior guess for the variance

    Returns
    -------
    mart, ArrayLike
        Hedged martingale (Waudby-Smith et al. (2023) https://arxiv.org/abs/2010.09686)
    """
    zeta_mu = zeta(mu, r)
    lambdas_plus = np.minimum(
        lambda_predmix_eb(
            z,
            alpha=alpha_opt,
            fixed_n=len(z),
            prior_mean=prior_mean,
            prior_variance=prior_variance,
        ),
        trunc_scale / zeta_mu,
    )
    lambdas_minus = np.minimum(
        lambda_predmix_eb(
            z,
            alpha=alpha_opt,
            fixed_n=len(z),
            prior_mean=prior_mean,
            prior_variance=prior_variance,
        ),
        trunc_scale / (1 - zeta_mu),
    )

    K_t_plus = np.cumprod(1 + lambdas_plus * (z - zeta_mu))
    K_t_minus = np.cumprod(1 - lambdas_minus * (z - zeta_mu))

    K_t = np.maximum(theta * K_t_plus, (1 - theta) * K_t_minus)

    return K_t


def nprr_hedged_ci(
    z: RealArray,
    r: float,
    alpha=0.1,
    trunc_scale: float = 4 / 5,
    theta: float = 1 / 2,
    prior_mean=1 / 2,
    prior_variance=1 / 8,
    breaks=1000,
    parallel=False,
) -> Tuple[RealArray, RealArray]:
    """
    Compute the NPRR-hedged confidence interval

    Parameters
    ----------
    z : RealArray
        The observed privatized data
    r : float
        The probability of outputting a true value in NPRR
    alpha : float, optional
        The desired type-I error level, by default 0.1
    trunc_scale : float, optional
        The truncation scale for the empirical Bernstein bound, by default 4/5
    theta : float, optional
        The mixing parameter for the hedged capital martingale, by default 1/2
    prior_mean : float, optional
        A prior guess for the true mean, by default 1/2
    prior_variance : float, optional
        A prior guess for the variance of the true mean, by default 1/8
    breaks : int, optional
        The number of breaks to use for the CS, by default 1000
    parallel : bool, optional
        Whether to use parallel processing, by default False

    Returns
    -------
    Tuple[RealArray, RealArray]
        The lower and upper bounds of the confidence interval
    """
    mart_fn = lambda z, mu: nprr_hedged_mart(
        z=z,
        mu=mu,
        r=r,
        theta=theta,
        trunc_scale=trunc_scale,
        alpha_opt=alpha / 2,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        l, u = cs_from_martingale(
            z, mart_fn=mart_fn, breaks=breaks, alpha=alpha, parallel=parallel
        )

    l = np.maximum.accumulate(l)
    u = np.minimum.accumulate(u)

    return l[-1], u[-1]


def nprr_gridKelly_mart(
    z: RealArray,
    mu: float,
    r: float,
    D: int,
    theta: float = 1 / 2,
) -> RealArray:
    """
    Compute the martingale for the grid Kelly strategy with D grid points

    Parameters
    ----------
    z : RealArray
        The sequence of private observations
    mu : float
        The mean to test against
    r : float
        The NPRR parameter determining the probability of outputting the stochastically rounded value.
    D : int
        The number of grid points to use in gridKelly
    theta : float, optional
        The proportion of wealth to use for positive bets, by default 1/2

    Returns
    -------
    RealArray
        The gridKelly martingale
    """
    zeta_mu = zeta(mu, r)
    lambdas_plus = [(d + 1) / ((D + 1) * zeta_mu) for d in range(D)]
    lambdas_minus = [(d + 1) / ((D + 1) * (1 - zeta_mu)) for d in range(D)]

    K_t_plus_list = [
        np.cumprod(1 + lambda_plus * (z - zeta_mu))
        for lambda_plus in lambdas_plus
    ]

    K_t_minus_list = [
        np.cumprod(1 - lambda_minus * (z - zeta_mu))
        for lambda_minus in lambdas_minus
    ]

    K_t_plus = np.array(reduce(add, K_t_plus_list)) / D
    K_t_minus = np.array(reduce(add, K_t_minus_list)) / D
    K_t = theta * K_t_plus + (1 - theta) * K_t_minus

    return K_t


def nprr_gridKelly_cs(
    z: RealArray,
    r: float,
    D: int,
    alpha=0.1,
    theta: float = 1 / 2,
    breaks=1000,
    parallel=False,
) -> Tuple[RealArray, RealArray]:
    """
    Compute the NPRR-gridKelly confidence sequence.

    Parameters
    ----------
    z : RealArray
        The sequence of private observations.
    r : float
        The NPRR parameter determining the probability of outputting the stochastically rounded value.
    D : int
        The number of grid points to use for gridKelly.
    alpha : float, optional
        The desired type-I error level.
    theta : float, optional
        The mixing parameter for gridKelly.
    breaks : int, optional
        The number of breaks to use for the confidence sequence.
    parallel : bool, optional
        Whether to use parallel processing.

    Returns
    -------
    l : RealArray
        The lower confidence sequence.
    u : RealArray
        The upper confidence sequence.
    """
    mart_fn = lambda z, mu: nprr_gridKelly_mart(
        z=z, mu=mu, r=r, D=D, theta=theta
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        l, u = cs_from_martingale(
            z, mart_fn=mart_fn, breaks=breaks, alpha=alpha, parallel=parallel
        )
    return l, u


def zeta(mu: float, r: float):
    """
    The mean of NPRR output given an original mean of mu

    Parameters
    ----------
    mu : float
        The original mean
    r : float
        The probability of outputting the stochastically rounded value in NPRR

    Returns
    -------
    float
        The mean of NPRR output
    """
    return r * mu + (1 - r) / 2


def rr_bernoulli_logLR_mart(
    z: RealArray,
    m: float,
    r: float,
    prior_m: float = 1 / 2,
    fake_obs: int = 1,
) -> RealArray:
    """
    Compute the log-likelihood ratio martingale given Bernoulli input

    Parameters
    ----------
    z : RealArray
        The sequence of private observations
    m : float
        The mean to test against
    r : float
        The NPRR parameter determining the probability of outputting the stochastically rounded value.
    prior_m : float, optional
        Prior guess for the mean, by default 1/2
    fake_obs : int, optional
        The number of fake observations to add to the prior, by default 1

    Returns
    -------
    RealArray
        The log-likelihood ratio martingale
    """
    assert all(np.isin(z, (0, 1)))

    S_t = np.cumsum(z)
    t = np.arange(1, len(z) + 1)
    zeta_hat = (prior_m * fake_obs + S_t) / (fake_obs + t)
    zeta_hat_prev = np.append(prior_m, zeta_hat[0 : (len(z) - 1)])

    log_numerator = binom.logpmf(z, 1, zeta_hat_prev)
    log_denominator = binom.logpmf(z, 1, zeta(m, r))
    log_mart = np.cumsum(log_numerator - log_denominator)

    return log_mart


def rr_bernoulli_LR_cs(
    z: RealArray,
    r: float,
    prior_m: float = 1 / 2,
    fake_obs: int = 1,
    alpha: float = 0.1,
    breaks: float = 1000,
    parallel: bool = False,
):
    """
    Confidence sequence for the log-likelihood ratio martingale given Bernoulli input

    Parameters
    ----------
    z : RealArray
        The sequence of privatized observations
    r : float
        The NPRR parameter determining the probability of outputting the stochastically rounded value.
    prior_m : float, optional
        Prior guess for the mean, by default 1/2
    fake_obs : int, optional
        The number of fake observations to add to the prior, by default 1
    alpha : float, optional
        The desired type-I error level, by default 0.1
    breaks : float, optional
        The number of breaks to use for the confidence sequence, by default 1000
    parallel : bool, optional
        Whether to use parallel processing, by default False

    Returns
    -------
    Tuple[RealArray, RealArray]
        The lower and upper confidence sequences
    """
    log_mart_fn = lambda x, m: rr_bernoulli_logLR_mart(
        z=x, m=m, r=r, prior_m=prior_m, fake_obs=fake_obs
    )

    l, u = cs_from_martingale(
        x=z,
        mart_fn=log_mart_fn,
        breaks=breaks,
        parallel=parallel,
        alpha=alpha,
        log_scale=True,
    )

    return l, u


def rr_bernoulli_conjmix_cs(
    z: RealArray,
    r: float,
    v_opt: float,
    alpha: float = 0.1,
    breaks: int = 1000,
    parallel: bool = False,
) -> Tuple[RealArray, RealArray]:
    """
    Conjugate mixture confidence sequence for the mean of Bernoulli input privatized via NPRR

    Parameters
    ----------
    z : ArrayLike
        The sequence of privatized observations
    r : float
        The NPRR parameter determining the probability of outputting the stochastically rounded value.
    v_opt : float
        The intrinsic time to optimize the bound for
    alpha : float, optional
        The desired type-I error level, by default 0.1
    breaks : int, optional
        The number of breaks to use for the confidence sequence, by default 1000
    parallel : bool, optional
        Whether to use parallel processing, by default False

    Returns
    -------
    Tuple[RealArray, RealArray]
        The lower and upper confidence sequences
    """
    t = np.arange(1, len(z) + 1)
    log_mart_fn = lambda x, m: beta_binomial_log_mixture(
        s=np.cumsum(z - zeta(m, r)),
        v=zeta(m, r) * (1 - zeta(m, r)) * t,
        v_opt=v_opt,
        g=zeta(m, r),
        h=1 - zeta(m, r),
        is_one_sided=False,
    )

    l, u = cs_from_martingale(
        x=z,
        mart_fn=log_mart_fn,
        breaks=breaks,
        alpha=alpha,
        parallel=parallel,
        log_scale=True,
    )

    return l, u
