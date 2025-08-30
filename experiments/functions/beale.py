import numpy as np

def beale_func(xy):
    """
    Beale test function (2 variables).

    f(x,y) = (1.5 - x + x*y)^2
           + (2.25 - x + x*y^2)^2
           + (2.625 - x + x*y^3)^2

    Domain usually: x,y in [-4.5, 4.5]
    Global minimum: f(3, 0.5) = 0
    """
    xy = np.asarray(xy, dtype=float)
    if xy.shape[-1] != 2:
        raise ValueError("Beale expects 2 variables: [x, y].")
    x, y = xy[..., 0], xy[..., 1]
    return ((1.5   - x + x*y)   ** 2
          + (2.25  - x + x*y**2)** 2
          + (2.625 - x + x*y**3)** 2)