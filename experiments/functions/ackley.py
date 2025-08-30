import numpy as np


def ackley_func(x):
    """Ackley function"""
    x = np.asarray(x)
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    return (
        -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
        - np.exp(sum_cos / n)
        + 20
        + np.e
    )

