import numpy as np


def griewank_func(x):
    """Griewank function"""
    x = np.asarray(x)
    sum_sq = np.sum(x**2) / 4000.0
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_sq - prod_cos + 1