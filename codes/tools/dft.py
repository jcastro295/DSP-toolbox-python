import numpy as np


def dft(xn, N=None):
    """
    Function to compute DFT
    This function computes the N-point DFT of a signal x(n)
    xn : Input signal in dicrete-time domain
    N: length of transformed signal X(k)
    """

    if not isinstance(xn, np.ndarray):
        xn = np.asarray(xn)

    L = len(xn)

    if N is None:
        N = L

    if N < L:
        xn = xn[:N]
    else:
        xn = np.concatenate([xn, np.zeros(N-L)])

    n = np.arange(N)[np.newaxis,:]      # row vector for n
    k = np.arange(N)[np.newaxis,:]      # row vecor for k

    WN = np.exp(-1j*2*np.pi/N)          # Wn factor
    nk = n.T @ k                        # creates a N by N matrix of nk values
    WNnk = WN**nk                       # DFT matrix

    Xk = xn @ WNnk                      # row vector for DFT coefficients

    return Xk

def dft_for_loop(xn, N=None):
    """
    Function to compute DFT
    This function computes the N-point DFT of a signal x(n)
    xn : Input signal in dicrete-time domain
    N: length of transformed signal X(k)
    """

    if not isinstance(xn, np.ndarray):
        xn = np.asarray(xn)

    L = len(xn)

    if N is None:
        N = L

    if N < L:
        x1 = xn[:N]
    else:
        x1 = np.concatenate([xn, np.zeros(N-L)])

    W = np.zeros((N,N), dtype=complex)
    for k in range(N): 
        for n in range(N): 
            W[k,n] = np.exp(-1j*2*np.pi*n*k/N) 

    Xk = W @ x1.T 

    return Xk
