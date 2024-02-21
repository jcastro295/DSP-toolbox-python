"""
quantization_error.py
---------------------

This script demonstrates how to plot the quantization error of a signal.
"""

import numpy as np
import matplotlib.pyplot as plt

from tools.quant_bits import quant_bits

# Signal parameters
fs = 1000  # sample rate (Hz)
T = 4  # duration (s)
n = np.arange(0, fs * T + 1) / fs  # time vector
f = 0.25  # frequency (Hz)
A = 0.5  # Max Amplitude 
N = 32  # bit-depth 

# Generate the original signal
x = A * np.cos(2 * np.pi * f * n)

# Quantize the signal
x_quantized, quantization_step = quant_bits(x, N)

# Calculate quantization error
quantization_error = x - x_quantized

# Plot quantization error and quantized signal
plt.figure(figsize=(8, 6))

q = 2/2**N
plt.plot(n, quantization_error, linewidth=0.5, color='#1f77b4')
plt.plot(n, x_quantized*q, color='#d62728')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Amplitude/quantization error', fontsize=12)
plt.title(f'Quantization error plot for {N}-bit representation', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks([-q / 2, -q / 4, 0, q / 4, q / 2],
           ['-q/2', '-q/4', '0', '+q/4', '+q/2'], fontsize=12)
plt.legend(['Quantization error', 'Discrete signal'], loc='lower right', fontsize=12)
plt.grid(True)
plt.show()
