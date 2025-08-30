import numpy as np

def himmelblau_func(xy):
    """
    Himmelblau's function (2 variables).

    f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2

    Domain: x,y in [-6, 6]
    Global minima (f=0) at (3,2), (-2.805118,3.131312),
    (-3.779310,-3.283186), (3.584428,-1.848126)
    """
    xy = np.asarray(xy, dtype=float)
    if xy.shape[-1] != 2:
        raise ValueError("Himmelblau expects 2 variables: [x, y].")
    x, y = xy[..., 0], xy[..., 1]
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2