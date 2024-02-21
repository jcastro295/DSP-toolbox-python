"""

stepseq.py
----------

"""

import numpy as np

def stepseq(n0, n1, n2):
    if n0 < n1 or n0 > n2 or n1 > n2:
        raise ValueError('arguments must satisfy n1 <= n0 <= n2')

    n = np.arange(n1, n2 + 1)
    x = (n - n0) >= 0
    return x.astype('uint8'), n
