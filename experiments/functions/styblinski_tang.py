import numpy as np

def styblinski_tang_func(x):
    """
    Styblinski–Tang function.

    f(x) = 0.5 * sum(x_i^4 - 16*x_i^2 + 5*x_i)

    Domain: x_i in [-5, 5]
    Global minimum: f(x) ≈ -39.16617 * n at x_i ≈ -2.903534
    """
    x = np.asarray(x, dtype=float)
    return 0.5 * np.sum(x**4 - 16*x**2 + 5*x)