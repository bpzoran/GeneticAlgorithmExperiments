import numpy as np


def rosenbrock_func(x):
    """Rosenbrock (Banana) function"""
    if isinstance(x, dict):
        x = list(x.values())
    x = np.asarray(x)
    rslt = np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (x[:-1] - 1.0) ** 2.0)
    return rslt