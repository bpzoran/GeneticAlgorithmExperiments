import numpy as np


def sphere_func(x):
    """
    Sphere test function.

    Parameters:
        x (array-like): Input vector of shape (n,)

    Returns:
        float: Function value
    """
    if isinstance(x, dict):
        x = list(x.values())
    x = np.asarray(x)
    return np.sum(x ** 2)