"""
spectrogram_signal_ex.py
------------------------

This script shows two examples of how to generate spectrograms using a
cosine and upsweep chirp (Low Frequency Modulated signal). Also, it
compares the hardcoded function with the matlab implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, spectrogram

from tools.spectrogram_coded import spectrogram_coded

fs = 200       # Hz, sampling frequency
fmin = 10      # Hz, minimum frequency to be saved
fmax = 100     # Hz, maximum freuqency to be saved
# Freq_spacing = fs/ns. 25k/4096 = 6.103 Hz
ns = 2**12     # number of samples for each FFT
n_secs = 5     # sec, total time interval to save from the beginning of the signal
N = fs*n_secs  # Number of time samples in x
t = np.arange(0,N)/fs

dt_save = ns/fs

# Calculate spectrograms 
t_window = 0.2                # Integration time in seconds, ideally it is ns/fs
n_window = fs*t_window         # Window size, i.e., number of points to be considered in the FFT
window = np.hamming(n_window) # Apply the window you would like to use.
overlap = 0.75                # Overlap (0.0 - 0.99).Typical value: 0.5

# The final time step will be: t_window*(1-overlap) in [sec]
# The frequency step is: fs/ns
n_overlap = np.round(n_window*overlap)   # Number of overlap points

# Noisy sine wave example:
omega = 2*np.pi*(2*fmin+fmax)/2
x = np.sin(omega*t) #+ 0.1*np.random.normal(0, .1, t.shape)

[P, F, T] = spectrogram_coded(x,window,n_overlap,ns,fs)
spec = 10*np.log10(np.abs(P)) # In dB
[F_scipy,T_scipy, P_scipy] = spectrogram(x, fs=fs, window=window, noverlap=n_overlap,
                             nfft=ns, detrend=False, mode='psd')
spec_scipy = 10*np.log10(np.abs(P_scipy)).T # In dB

# The output spectrogram contains all the possible frequencies.
# If you want to select a certain frequency band you can index the vector 'F' and matrix 'spec' :
f_idx = np.argwhere((F > fmin) & (F < fmax))
F = np.squeeze(F[f_idx]) # new list of frequencies
spec = np.squeeze(spec[:,f_idx]) # band passed spectrogram
F_scipy = np.squeeze(F_scipy[f_idx]) # new list of frequencies
spec_scipy = np.squeeze(spec_scipy[:,f_idx]).T # band passed spectrogram

fig = plt.figure(figsize=(15, 6))

ax1 = fig.add_subplot(1,3,1)
ax1.plot(t, x)
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Pressure (Pa)', fontsize=12)
ax1.set_title('Time Series', fontsize=14)
ax1.set_xlim([t[0], t[-1]])

# Plot spectrogram snapshots (ns samples spaced dt_save apart)
ax2 = fig.add_subplot(1,3,2)
im = ax2.pcolormesh(T, F, spec.T, shading='auto')
cbar = plt.colorbar(im, ax=ax2)
cbar.set_label('Power Spectral Density (dB )')
ax2.set_xlabel('Time (sec)', fontsize=12)
ax2.set_ylabel('Frequency (Hz)', fontsize=12)
ax2.set_title(f'Spectroogram (coded): {len(F)} freqs x {len(T)} times', fontsize=14)

ax3 = fig.add_subplot(1,3,3)
im = ax3.pcolormesh(T_scipy, F_scipy, spec_scipy, shading='auto')
cbar = plt.colorbar(im, ax=ax3)
cbar.set_label('Power Spectral Density (dB )')
ax3.set_xlabel('Time (sec)', fontsize=12)
ax3.set_ylabel('Frequency (Hz)', fontsize=12)
ax3.set_title(f'Spectroogram (scipy): {len(F)} freqs x {len(T)} times', fontsize=14)

fig.tight_layout()

# second example
# Noisy chirp example
x = chirp(t, f0=fmin, f1=fmax, t1=n_secs, method='linear') + 0.2*np.random.normal(0, .1, t.shape)

[s, F, T] = spectrogram_coded(x,window,n_overlap,ns,fs)
spec = 10*np.log10(np.abs(s)) # In dB
[F_scipy,T_scipy, P_scipy] = spectrogram(x, fs=fs, window=window, noverlap=n_overlap,
                             nfft=ns, detrend=False, mode='psd')
spec_scipy = 10*np.log10(np.abs(P_scipy)).T # In dB

# The output spectrogram contains all the possible frequencies.
# If you want to select a certain frequency band you can index the vector 'F' and matrix 'spec' :
f_idx = np.argwhere((F > fmin) & (F < fmax))
F = np.squeeze(F[f_idx]) # new list of frequencies
spec = np.squeeze(spec[:,f_idx]) # band passed spectrogram
F_scipy = np.squeeze(F_scipy[f_idx]) # new list of frequencies
spec_scipy = np.squeeze(spec_scipy[:,f_idx]).T # band passed spectrogram

fig = plt.figure(figsize=(15, 6))

ax1 = fig.add_subplot(1,3,1)
ax1.plot(t, x)
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Pressure (Pa)', fontsize=12)
ax1.set_title('Time Series', fontsize=14)
ax1.set_xlim([t[0], t[-1]])

# Plot spectrogram snapshots (ns samples spaced dt_save apart)
ax2 = fig.add_subplot(1,3,2)
im = ax2.pcolormesh(T, F, spec.T, shading='auto')
cbar = plt.colorbar(im, ax=ax2)
cbar.set_label('Power Spectral Density (dB )')
ax2.set_xlabel('Time (sec)', fontsize=12)
ax2.set_ylabel('Frequency (Hz)', fontsize=12)
ax2.set_title(f'Spectroogram (coded): {len(F)} freqs x {len(T)} times', fontsize=14)

ax3 = fig.add_subplot(1,3,3)
im = ax3.pcolormesh(T_scipy, F_scipy, spec_scipy, shading='auto')
cbar = plt.colorbar(im, ax=ax3)
cbar.set_label('Power Spectral Density (dB )')
ax3.set_xlabel('Time (sec)', fontsize=12)
ax3.set_ylabel('Frequency (Hz)', fontsize=12)
ax3.set_title(f'Spectroogram (scipy): {len(F)} freqs x {len(T)} times', fontsize=14)

fig.tight_layout()

plt.show()