"""
compute_fourier_cofficients_audio.py
------------------------------------

This script computes the Fourier coefficients of an audio signal and
plots the magnitude and phase angle spectra. It also plays the original
signal and signals with different numbers of Fourier components.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import resample_poly

from tools.compute_complex_fourier_coef import compute_complex_fourier_coef


data_folder = '../data'

# Load audio file
fs, y = wavfile.read(os.path.join(data_folder, 'ahh_sound.wav'))
y = y / 2**np.ceil(np.log2(y.max())) # Normalize to [-1, 1]

# Resample if needed
resampling_fs = 5000
#y = resample(y, int(len(y) * resampling_fs / fs))
y = resample_poly(y, resampling_fs, fs)
fs = resampling_fs

t = np.arange(len(y)) / fs

# Compute Fourier coefficients
n_coefficients = 3000
f, coeffs = compute_complex_fourier_coef(y, fs, n_coefficients)

# Plot magnitude spectrum
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.stem(f, np.abs(coeffs), linefmt='b-', markerfmt='bo', basefmt=" ")
plt.title('(a) Magnitude spectrum', fontsize=14)
plt.ylabel('|a_k|', fontsize=12)
plt.xlim([np.min(f), np.max(f)])

# Plot phase angle spectrum
plt.subplot(2, 1, 2)
plt.stem(f, np.angle(coeffs), linefmt='b-', markerfmt='bo', basefmt=" ")
plt.title('(b) Phase angle spectrum (rad)', fontsize=14)
plt.xlabel('Frequency f (Hz)', fontsize=12)
plt.ylabel('angle(a_k) (rad)', fontsize=12)
plt.xlim([np.min(f), np.max(f)])

plt.tight_layout()

# Second figure
fig = plt.figure(figsize=(10, 6))

ax = fig.add_subplot(4, 1, 1)
ax.plot(t, y, linewidth=1.5)
ax.set_title('Original signal', fontsize=14)
ax.set_ylabel('Amplitude', fontsize=12)
ax.set_ylim([-1.2, 1.2])
sd.play(y/max(abs(y)), fs)
print('Original signal played. Type any key to continue...')
input()


# Play signals with different numbers of Fourier components
for i, num_components in enumerate([len(coeffs) // 4, len(coeffs) // 2, len(coeffs)], start=2):

    ax = fig.add_subplot(4, 1, i)

    idx = np.arange(0, num_components)
    y_comp = np.real(np.sum(np.exp(2j * np.pi * t.reshape(-1, 1) * f[idx]) * coeffs[idx], axis=1))

    ax.plot(t, y_comp, linewidth=1.5)
    ax.set_title(f'Signal with {num_components // 2 + 1} Fourier components', fontsize=14)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_ylim([-1.2, 1.2])

    # Play the signal
    sd.play(y/max(abs(y_comp)), fs)
    print(f'Signal with {num_components // 2 + 1} coefficients played. Press Enter to continue...')
    fig.canvas.draw_idle()
    input()


plt.xlabel('Time (s)', fontsize=12)
plt.tight_layout()
plt.show()
