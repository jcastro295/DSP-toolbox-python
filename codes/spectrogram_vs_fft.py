"""
spectrogram_vs_fft.py
---------------------

This code compares the spectrogram (time-frequency) representation vs the
FFT (frequency-only) represention
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

fs = 100
N = 2000
n = np.arange(N)/fs
freq = fs/N*np.arange(-N/2,N/2)

x1 = np.sin(2*2*np.pi*n)
x2 = np.sin(4*2*np.pi*n)

x = np.concatenate([x1[n<=10], x2[n>10]])

x_flipped = x[::-1]

X = np.fft.fft(x)
X_flipped = np.fft.fft(x)

# spectrogram parameters 
t_window = 2                   #  Integration time in seconds, ideally it is ns/fs
n_window = fs*t_window         #  Window size, i.e., number of points to be considered in the FFT
window = np.hamming(n_window)  #  Apply the window you would like to use.
overlap = 0.9                  #  Overlap (0.0 - 0.99).Typical value: 0.5
ns = n_window                  #  number of samples for each FFT

# The final time step will be: t_window*(1-overlap) in [sec]
# The frequency step is: fs/ns
n_overlap = round(n_window*overlap)   # Number of overlap points


fig = plt.figure(figsize=(10, 8), layout='constrained')

ax1 = fig.add_subplot(3,2,1)
ax1.plot(n,x)
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.set_title('Time series x_1(n)', fontsize=14)

ax2 = fig.add_subplot(3,2,2)
ax2.plot(freq, x_flipped)
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.set_title('Time series x_2(n)', fontsize=14)

ax3 = fig.add_subplot(3,2,3)
ax3.plot(n,abs(np.fft.fftshift(X)))
ax3.set_xlabel('Frequency (Hz)', fontsize=12)
ax3.set_ylabel('Amplitude', fontsize=12)
ax3.set_title('Magnitude FFT X_1(k)', fontsize=14)

ax4 = fig.add_subplot(3,2,4)
ax4.plot(freq,abs(np.fft.fftshift(X_flipped)))
ax4.set_xlabel('Frequency (Hz)', fontsize=12)
ax4.set_ylabel('Amplitude', fontsize=12)
ax4.set_title('Magnitude FFT X_2(k)', fontsize=14)


[F, T, P] = spectrogram(x, fs=fs, window=window, noverlap=n_overlap,
                       nfft=ns, detrend=False, mode='psd')
spec = 10*np.log10(abs(P)).T #  In dB

[F_flipped, T_flipped, P_flipped] = spectrogram(x_flipped, fs=fs, window=window, noverlap=n_overlap,
                       nfft=ns, detrend=False, mode='psd')
spec_flipped = 10*np.log10(abs(P_flipped)).T #  In dB

ax5 = fig.add_subplot(3,2,5)
im1 = ax5.pcolormesh(T, F, spec.T, shading='auto', vmin=-40, vmax=0, cmap='jet')
cbar = plt.colorbar(im1, ax=ax5, label='PSD (dB)', pad=0.01)
ax5.set_xlabel('Time (sec)', fontsize=12)
ax5.set_ylabel('Frequency (Hz)', fontsize=12)
ax5.set_title('Spectrogram x_1(n)', fontsize=14)


ax6 = fig.add_subplot(3,2,6)
im2 = ax6.pcolormesh(T_flipped, F_flipped, spec_flipped.T, shading='auto', vmin=-40, vmax=0, cmap='jet')
cbar = plt.colorbar(im2, ax=ax6, label='PSD (dB)', pad=0.01)
ax6.set_xlabel('Time (sec)', fontsize=12)
ax6.set_ylabel('Frequency (Hz)', fontsize=12)
ax6.set_title('Spectrogram x_2(n)', fontsize=14)


plt.show()