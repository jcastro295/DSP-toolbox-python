import numpy as np

def idft(Xk, N=None):
    """
    function to calculate IDFT 
	This function computes the N-point IDFT of a signal X(k)
    Xk : Input signal in discrete-frequency domain
    N: length of transformed signal x(n)
    """

    if not isinstance(Xk, np.ndarray):
        Xk = np.asarray(Xk)

    L = len(Xk); 

    if N is None:
        N = L

    if N < L:
        Xk = Xk[:N]
    else:
        Xk = np.concatenate([Xk, np.zeros(N-L)])

    n = np.arange(N)[np.newaxis,:]   # row vector for n
    k = np.arange(N)[np.newaxis,:]   # row vecor for k

    WN = np.exp(-1j*2*np.pi/N)       # Wn factor
    nk = n.T @ k;                    # creates a N by N matrix of nk values
    WNnk = WN**(-nk);                # IDFT matrix

    xn = (Xk @ WNnk)/N;              # row vector for IDFT values

    return xn


def idft_for_loop(Xk, N=None):
    """
    function to calculate IDFT 
	This function computes the N-point IDFT of a signal X(k)
    Xk : Input signal in discrete-frequency domain
    N: length of transformed signal x(n)
    """

    if not isinstance(Xk, np.ndarray):
        Xk = np.asarray(Xk)

    L = len(Xk); 

    if N is None:
        N = L

    if N < L:
        X1 = Xk[:N]
    else:
        X1 = np.concatenate([Xk, np.zeros(N-L)])

    W = np.zeros((N,N), dtype=complex)
    for k in range(N):
        for n in range(N): 
            W[k,n] = np.exp(1j*2*np.pi*n*k/N)

    xn = (W.T @ X1.T)/N

    return xn
