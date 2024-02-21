"""
psd_signal_ex.py
----------------

This code computes the power spectral density of a cosine signal with
white noise. For further details:
https://www.mathworks.com/help/signal/ug/power-spectral-density-estimates-using-fft.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from scipy.signal.windows import boxcar


fs = 1000
t = np.arange(0,1,1/fs)
x = np.cos(2*np.pi*100*t) + 0.1*np.random.randn(len(t))

N = len(x)
xdft = np.fft.fft(x)
xdft = xdft[:int(N/2)+1]
psdx = (1/(fs*N)) * np.abs(xdft)**2
psdx[1:-1] = 2*psdx[1:-1]
freq = np.arange(0, fs/2+fs/len(x), fs/len(x))

fig = plt.figure(figsize=(14, 8))

ax1 = fig.add_subplot(2, 2, (1, 3))
ax1.plot(t,x)
ax1.grid(True)
ax1.set_title('Original signal x(n)', fontsize=14)
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.set_xlim([t[0], t[-1]])

ax2 = fig.add_subplot(2,2,2)
ax2.plot(freq, np.abs(xdft))
ax2.grid(True)
ax2.set_title('Magnitude of FFT X(k)', fontsize=14)
ax2.set_xlabel('Frequency (Hz)', fontsize=12)
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.set_xlim([freq[0], freq[-1]])

ax3 = fig.add_subplot(2,2,4)
ax3.plot(freq, 20*np.log10(psdx))
ax3.grid(True)
ax3.set_title('Power Spectral Density Using FFT', fontsize=14)
ax3.set_xlabel('Frequency (Hz)', fontsize=12)
ax3.set_ylabel('Power/Frequency (dB/Hz)', fontsize=12)
ax3.set_xlim([freq[0], freq[-1]])

fig.tight_layout()


# Scipy supports a function 
[f, pxx] = periodogram(x, fs, boxcar(N), N, detrend=False)

fig = plt.figure(figsize=(10, 8))

ax1 = fig.add_subplot(2,1,1)
ax1.plot(freq, 20*np.log10(psdx))
ax1.grid(True)
ax1.set_title('Power Spectral Density Using FFT', fontsize=14)
ax1.set_xlabel('Frequency (Hz)', fontsize=12)
ax1.set_ylabel('Power/Frequency (dB/Hz)', fontsize=12)
ax1.set_xlim([freq[0], freq[-1]])

ax2 = fig.add_subplot(2,1,2)
ax2.plot(f, 20*np.log10(pxx))
ax2.grid(True)
ax2.set_title('Power Spectral Density using Scipy function', fontsize=14)
ax2.set_xlabel('Frequency (Hz)', fontsize=12)
ax2.set_ylabel('Power/Frequency (dB/Hz)', fontsize=12)
ax2.set_xlim([f[0], f[-1]])

# compute the error (difference) between FFT-based implementation and
# periodogram
mxerr = np.max(psdx.T-pxx)
print(f'The error between both implementations is {mxerr:2.5e}')

fig.tight_layout()
plt.show()