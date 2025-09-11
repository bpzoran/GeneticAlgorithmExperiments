import numpy as np

def booth_func(xy) -> float:
    """
    Booth function.
    Global minimum at (x, y) = (1, 3) with f = 0.
    Works with scalars or numpy arrays.
    f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2
    """
    xy = np.asarray(xy, dtype=float)
    if xy.shape[-1] != 2:
        raise ValueError("Beale expects 2 variables: [x, y].")
    x, y = xy[..., 0], xy[..., 1]
    return (x + 2*y - 7)**2 + (2*x + y - 5)**2