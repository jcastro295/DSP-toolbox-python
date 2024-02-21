"""

evenodd.py
----------

"""

import numpy as np

def evenodd(x, n):
    if any(np.imag(x) != 0):
        raise ValueError('x is not a real sequence')

    m = -np.flip(n)
    m1 = min([m.min(), n.min()])
    m2 = max([m.max(), n.max()])
    m = np.arange(m1, m2 + 1)

    nm = n[0] - m[0]
    n1 = np.arange(len(n))
    x1 = np.zeros(len(m))

    x1[n1 + nm] = x
    x = x1

    xe = 0.5 * (x + np.flip(x))
    xo = 0.5 * (x - np.flip(x))

    return xe, xo, m
