"""

ifft.py
-------

"""

import numpy as np

def ifft_core(x):
    """
    A recursive implementation of 
    the 1D Cooley-Tukey IFFT, the 
    input should have a length of 
    power of 2. 
    """

    N = len(x)

    if N == 1:
        return x
    else:
        x_even = ifft_core(x[::2])  # recursive call for even-indexed elements of x
        x_odd = ifft_core(x[1::2])  # recursive call for odd-indexed elements of x
        Wn = np.exp(2j*np.pi*np.arange(N)/ N)

        xn = np.concatenate([x_even + Wn[:int(N/2)]*x_odd,
                             x_even + Wn[int(N/2):]*x_odd])
        return xn
    

def ifft_coded(x):
    return ifft_core(x)/len(x)