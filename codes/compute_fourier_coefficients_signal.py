"""
compute_fourier_coefficients_signal.py
--------------------------------------

This script computes the Fourier coefficients of a squared signal and 
plots the magnitude and phase angle spectra.
"""

import numpy as np
import matplotlib.pyplot as plt

from tools.compute_complex_fourier_coef import compute_complex_fourier_coef


fs = 5  # Sampling frequency
t = np.arange(0, 6 * fs * np.pi) / fs  # Time vector
n_coefficients = 10  # Number of Fourier coefficients

# Generate squared signal
y = np.sign(np.sin(t))

# Compute Fourier coefficients
f, coeffs = compute_complex_fourier_coef(y, fs, n_coefficients)

# Plot magnitude spectrum
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.stem(f, np.abs(coeffs), linefmt='b-', markerfmt='bo')
plt.title('(a) Magnitude spectrum', fontsize=14)
plt.ylabel('|a_k|', fontsize=12)
plt.xlim([np.min(f), np.max(f)])

# Plot phase angle spectrum
plt.subplot(2, 1, 2)
plt.stem(f, np.angle(coeffs), linefmt='b-', markerfmt='bo')
plt.title('(b) Phase angle spectrum (rad)', fontsize=14)
plt.xlabel('Frequency f (Hz)', fontsize=12)
plt.ylabel('angle(a_k) (rad)', fontsize=12)
plt.xlim([np.min(f), np.max(f)])

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(8, 10))
ax = fig.add_subplot(4, 1, 1)
ax.plot(t/np.pi, y, linewidth=1.5)
ax.set_title('Original signal', fontsize=14)
ax.set_ylabel('Amplitude', fontsize=12)
ax.set_ylim([-1.2, 1.2])

# Generate signals with different numbers of Fourier components
for i, num_components in enumerate([len(coeffs) // 4, len(coeffs) // 2, len(coeffs)], start=2):
    
    ax = fig.add_subplot(4, 1, i)

    idx = np.arange(0, num_components)
    y_comp = np.real(np.sum(np.exp(2j * np.pi * t.reshape(-1, 1) * f[idx]) * coeffs[idx], axis=1))

    ax.plot(t/np.pi, y_comp, linewidth=1.5)
    ax.set_title(f'Signal with {num_components // 2 + 1} Fourier components', fontsize=14)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_ylim([-1.2, 1.2])

plt.xlabel('Time (t/pi)', fontsize=12)
plt.tight_layout()
plt.show()
