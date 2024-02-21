"""

fft.py
------

"""

import numpy as np


def fft_coded(x):
    """
    A recursive implementation of 
    the 1D Cooley-Tukey FFT, the 
    input should have a length of 
    power of 2. 
    """

    N = len(x)

    if N == 1:
        return x
    else:
        X_even = fft_coded(x[::2])  # recursive call for even-indexed elements of x
        X_odd = fft_coded(x[1::2])  # recursive call for odd-indexed elements of x
        Wn = np.exp(-2j*np.pi*np.arange(N)/ N)

        Xk = np.concatenate([X_even+Wn[:int(N/2)]*X_odd,
                            X_even+Wn[int(N/2):]*X_odd])
        return Xk
