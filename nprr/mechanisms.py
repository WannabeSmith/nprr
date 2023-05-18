"""
Privacy mechanisms and utilities for working with them.
"""
from typing import Tuple
from numpy.typing import NDArray
import warnings
import numpy as np
from scipy.optimize import minimize, Bounds

from nprr.utils import RealArray

def laplace(x: NDArray[np.float_], eps: float) -> NDArray[np.float_]:
    """
    Laplace mechanism

    Parameters
    ----------
    x, RealArray
        Input data to privatize
    eps, float
        Privacy parameter

    Returns
    -------
    x + noise, RealArray
    """
    return x + np.random.laplace(loc=0, scale=1 / eps, size=len(x))


def warner_rr(x: np.ndarray, r: float) -> np.ndarray:
    """
    Warner's randomized response

    Parameters
    ----------
    x, RealArray
        Input data to privatize
    r, float
        Probability of responding truthfully

    Returns
    -------
    z, RealArray
        Privatized view of x
    """
    assert all(np.logical_or(x == 1, x == 0))

    indicator_randomized = np.random.binomial(1, 1 - r, len(x)) == 1
    z = np.where(indicator_randomized, np.random.binomial(1, 0.5, len(x)), x)
    return z


def multinomial_rr(x: np.ndarray, K: int, r: float) -> np.ndarray:
    """
    Multivariate generalization of Warner's randomized response

    Parameters
    ----------
    x, RealArray
        Input data to privatize
    K, int
        Number of categories
    r, float
        Probability of responding truthfully

    Returns
    -------
    z, RealArray
        Privatized view of x
    """
    # check that x contains K-vectors
    assert all(np.isin(x, range(K)))

    indicator_randomized = np.random.binomial(1, 1 - r, len(x)) == 1
    random_multinomial = np.random.random_integers(low=0, high=K)
    z = np.where(indicator_randomized, random_multinomial, x)

    return z


def mu_hat_regularized(z, r) -> RealArray:
    """
    Regularized estimate of the mean given NPRR-privatized data

    Parameters
    ----------
    z, RealArray
        Privatized data
    r, float
        Probability of outputting stochastically rounded data, given in NPRR

    Returns
    -------
    mu_hat, RealArray
        Regularized estimate of the mean
    """
    if isinstance(r, float) or isinstance(r, int):
        r = np.repeat(r, len(z))

    return (1 / 2 + np.cumsum((z - (1 - r) / 2))) / (1 + np.cumsum(r))


def nprr(x: NDArray[np.float_], r: float, G: int) -> NDArray[np.float_]:
    """
    Nonparametric randomized response mechanism

    Parameters
    ----------
    x, RealArray
        Input data to privatize
    r, float
        Probability of outputting stochastically rounded data
    G, int
        Number of bins to use for stochastic rounding

    Returns
    -------
    z, RealArray
        Privatized view of x
    """
    # check that x is in [0, 1]
    assert all(x >= 0) and all(x <= 1)
    x_grid_floor = np.floor(np.multiply(x, G)) / G
    x_grid_ceil = np.ceil(np.multiply(x, G)) / G

    prob_ceil = G * (x - x_grid_floor)
    indicator_ceil = np.random.binomial(1, prob_ceil, len(x))
    x_discretized = x_grid_ceil * indicator_ceil + x_grid_floor * (
        1 - indicator_ceil
    )

    indicator_discretized_x = np.random.binomial(1, r, len(x))

    discrete_uniform = np.round(np.random.beta(1, 1, len(x)) * G) / G
    z = x_discretized * indicator_discretized_x + discrete_uniform * (
        1 - indicator_discretized_x
    )

    return z


def pmf_approx(r: float, G: float, ceil_or_floor: bool):
    """
    Approximate probability mass function of NPRR output

    Parameters
    ----------
    r, float
        Probability of outputting stochastically rounded data
    G, float
        Number of bins to use for stochastic rounding
    ceil_or_floor, bool
        Whether the input is equal to the ceil or floor (i.e. unchanged input after stochastic rounding)

    Returns
    -------
    pmf, float
        Approximate probability mass function of NPRR output
    """
    if ceil_or_floor:
        return (1 - r) / (G + 1) + r / 2
    else:
        return (1 - r) / (G + 1)


def uniform_nprr_entropy(r: float, G: float):
    """
    Entropy of NPRR output from uniform input

    Parameters
    ----------
    r, float
        Probability of outputting stochastically rounded data
    G, float
        Number of bins to use for stochastic rounding

    Returns
    -------
    entropy, float
        Entropy of NPRR output from uniform input
    """
    pmf_not_cf = pmf_approx(r, G, ceil_or_floor=False)
    pmf_cf = pmf_approx(r, G, ceil_or_floor=True)
    return (G - 1) * pmf_not_cf * np.log2(pmf_not_cf) + 2 * pmf_cf * np.log2(
        pmf_cf
    )


def G_from_r_eps(r: float, eps: float):
    """
    Number of bins to use for stochastic rounding to achieve a given epsilon given r

    Parameters
    ----------
    r, float
        r value in NPRR
    eps, float
        Desired privacy level

    Returns
    -------
    G, float
        Number of bins to use for stochastic rounding
    """
    return (np.exp(eps) - 1) * (1 - r) / r - 1


def r_from_G_eps(G: float, eps: float):
    """
    r value in NPRR to achieve a given epsilon given G

    Parameters
    ----------
    G, float
        Number of bins used for stochastic rounding
    eps, float
        Desired privacy level

    Returns
    -------
    r, float
        Probability of outputting (stochastically rounded) input data
    """
    return (np.exp(eps) - 1) / (np.exp(eps) + G)


def r_G_opt_entropy(eps: float) -> Tuple[float, int]:
    """
    Optimal r and G values for NPRR to achieve a given epsilon using the heuristic entropy method

    Parameters
    ----------
    eps, float
        Desired privacy level

    Returns
    -------
    r_opt, float
        Optimal r value in NPRR
    G_opt, int
        Optimal number of bins to use for stochastic rounding
    """
    def objective(r):
        """
        Objective function for optimization
        """
        return uniform_nprr_entropy(r, G_from_r_eps(r, eps))

    bounds = Bounds(0.51, 0.99)
    minimization = minimize(
        objective, x0=(0.6,), bounds=bounds, method="L-BFGS-B"
    )

    r_opt = minimization["x"]

    # Find implicit value of G, truncate below by 1
    G_opt = max(1, G_from_r_eps(r_opt, eps))

    # Discretize based on whether ceil or floor is better.
    G_floor = np.floor(G_opt)
    G_ceil = np.ceil(G_opt)
    G_opt_discrete = (
        G_floor
        if objective(r_from_G_eps(G_floor, eps)) < objective(r_from_G_eps(G_ceil, eps))
        else G_ceil
    )

    r_opt_adjusted = (np.exp(eps) - 1) / (G_opt_discrete + np.exp(eps))

    eps_final = np.log(
        1 + (G_opt_discrete + 1) * r_opt_adjusted / (1 - r_opt_adjusted)
    )

    if not np.isclose(eps, eps_final):
        warnings.warn(
            "Unable to satisfy "
            + str(eps)
            + "-LDP exactly. Instead, eps = "
            + str(eps_final)
            + "."
        )

    return float(r_opt_adjusted), int(G_opt_discrete)

