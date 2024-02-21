"""

diracseq.py
------------

"""

import numpy as np

def diracseq(n, a):
    return np.array([1 if i == a else 0 for i in n])