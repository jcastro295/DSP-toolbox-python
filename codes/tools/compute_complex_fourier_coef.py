"""

compute_complex_fourier_coef.py
-------------------------------

"""

import numpy as np

def compute_complex_fourier_coef(y, fs, n_modes):
    tc = 1 / fs * np.arange(len(y))
    L = tc[-1] / 2  # The domain is assumed to be 2L
    nvec = np.arange(-n_modes, n_modes + 1)
    freq = nvec / (2 * L)  # Frequencies

    # Estimate coefficients using trapezoid rule
    c = np.zeros(len(nvec), dtype=np.complex128)
    for k in range(len(nvec)):
        c[k] = (1 / (2 * L)) * np.trapz(np.exp(-1j * nvec[k] * np.pi / L * tc) * y, tc)
        
    return freq, c
