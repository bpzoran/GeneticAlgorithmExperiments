import numpy as np


def rastrigin_func(x):
    """Rastrigin function"""
    x = np.asarray(x)
    n = len(x)
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))