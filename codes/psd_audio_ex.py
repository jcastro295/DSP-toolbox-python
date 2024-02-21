"""
psd_audio_ex.py
----------------

This script computes the power spectral density (PSD) of an audio signal using
the FFT-based and periodogram-based methods. The audio signal is read from a
.wav file and is normalized to the range [-1, 1]. The PSD is computed using the
FFT-based method and the periodogram-based method. The PSDs are plotted and
compared. The audio signal is also played.
For further details: 
https://www.mathworks.com/help/signal/ug/power-spectral-density-estimates-using-fft.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import periodogram
from scipy.signal.windows import boxcar
import scipy.io.wavfile as wavfile


data_folder = 'data'

# Read the audio file
fs, x = wavfile.read(os.path.join(data_folder, 'flute.wav'))
x = x / 2**np.ceil(np.log2(x.max())) # Normalize to [-1, 1]

# Crop the signal
x = x[:int(0.23 * len(x))]
# select only one channel
x = x[:, 0]

N = len(x)
t = np.arange(0, len(x)) / fs
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

# play the sound
sd.play(x, fs)

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
ax2.set_title('Power Spectral Density Using Scipy function', fontsize=14)
ax2.set_xlabel('Frequency (Hz)', fontsize=12)
ax2.set_ylabel('Power/Frequency (dB/Hz)', fontsize=12)
ax2.set_xlim([f[0], f[-1]])

# compute the error (difference) between FFT-based implementation and
# periodogram
mxerr = np.max(psdx.T-pxx)
print(f'The error between both implementations is {mxerr:2.5e}')

fig.tight_layout()

plt.show()