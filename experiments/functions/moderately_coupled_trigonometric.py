import numpy as np


def moderately_coupled_trigonometric_func(args):
    """
    Moderately coupled trigonometric function computes a mathematical expression using trigonometric
    and arithmetic operations on elements of an input array.

    This function performs the following steps:

    1. Computes the sine of the first element (args[0]).
    2. Computes the cosine of the second element (args[1]).
    3. Adds the square of the third element (args[2]).
    4. Multiplies the sine of the fourth element (args[3]) by the cosine of the fifth element (args[4]) and adds the result.
    5. Adds the value of the sixth element (args[5]).
    6. Multiplies the cosine of the seventh element (args[6]) by the eighth element (args[7]) and adds the result.

    The function ensures that the input list contains exactly 8 elements. If not, it raises a ValueError.

    Each operation in this function is independent of the others, but each can contain one or more local minima,
    making the function useful for optimization experiments.

    Parameters:
    args (list): A list of 8 numerical values used as input for the function.

    Returns:
    float: The result of the mathematical expression.
    """
    if len(args) != 8:
        raise ValueError("Input vector must contain 8 variables.")
    return np.sin(args[0]) + np.cos(args[1]) + args[2] ** 2 + np.sin(args[3]) * np.cos(args[4]) + args[5] + np.cos(
        args[6]) * args[7]