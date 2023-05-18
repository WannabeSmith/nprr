"""
CGF-like functions for various distributions.
"""
import numpy as np


def psi_H(l):
    """
    CGF upper-bound for [0, 1]-bounded random variables.
    """
    return psi_N(l, scale=1 / 2)


def psi_N(l, scale):
    """
    CGF for Gaussian random variables.
    """
    return np.power(scale, 2) * np.power(l, 2) / 2


def psi_E(l):
    """
    CGF for exponential random variables.
    """
    return (-np.log(1 - l) - l) / 4


def psi_B(l, p):
    """
    CGF for Bernoulli random variables.
    """
    return np.log(1 - p + p * np.exp(l))


def psi_L(l, scale):
    """
    CGF for Laplace random variables.
    """
    return -np.log(1 - np.power(scale, 2) * np.power(l, 2))
