"""

conv_m.py
---------

"""

import numpy as np

def conv_m(x, nx, h, nh):
    """
    Modified convolution routine for signal processing

    Parameters:
    x (array): First signal on support nx
    nx (array): Support of x
    h (array): Second signal on support nh
    nh (array): Support of h

    Returns:
    y (array): Convolution result
    ny (array): Support of y
    """

    nyb = nx[0] + nh[0]
    nye = nx[-1] + nh[-1]
    ny = np.arange(nyb, nye + 1)
    y = np.convolve(x, h, mode='full')

    return y, ny

