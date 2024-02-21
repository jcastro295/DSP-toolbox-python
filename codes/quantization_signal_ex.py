"""
quantization_signal_ex.py
-------------------------

This script demonstrates how to plot the original, downsampled, and quantized
versions of a signal.
"""

import numpy as np
import matplotlib.pyplot as plt

from tools.quant_bits import quant_bits

# Signal parameters
fs = 1000  # sample rate (Hz)
T = 4  # duration (s)
n = np.arange(0, fs * T) / fs  # time vector
f = 0.25  # frequency (Hz)
N = 4  # bit-depth
A = 0.5
f_sampled = 20 * f  # downsample frequency

# Generate original signal
x = A * np.cos(2 * np.pi * f * n)

# Downsample the signal
x_sampled = x[::int(fs / f_sampled)]
n_sampled = n[::int(fs / f_sampled)]

# Quantize the signal
x_quantized, _ = quant_bits(x_sampled, N)

# Plot the signals
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(n, x, color='#7f7f7f')
plt.title('Original signal')
plt.xlabel('Time (t)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(1, 3, 2)
_, stem1, _ = plt.stem(n_sampled, x_sampled, linefmt='-', markerfmt='o')
plt.setp(stem1, color='#1f77b4') 
plt.title('Subsampled signal')
plt.xlabel('Time (nT)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(1, 3, 3)
marker2, stem2, _ = plt.stem(n_sampled, x_quantized, linefmt='--', markerfmt='D')
plt.setp(stem2, color='#d62728') 
plt.setp(marker2, markerfacecolor='#d62728', markeredgecolor='#d62728') 
plt.title(f'{N}-bit quantized version of x')
plt.xlabel('Time (nT)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()
