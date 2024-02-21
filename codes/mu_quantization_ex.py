"""
mu_quantization_ex.py
---------------------

This script demonstrates the quantization of an audio signal using linear and
mu-law companding.
"""

import os
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

from tools.quant_bits import quant_bits


data_folder = 'data'

# Parameters
u = 255  # mu-law compression factor
N = 4    # bit-depth

# Read the audio file
fs, y = wavfile.read(os.path.join(data_folder, 'flute.wav'))
y = y / 2**np.ceil(np.log2(y.max())) # Normalize to [-1, 1]

# Crop the signal
y = y[:int(0.23 * len(y))]

n = np.arange(0, len(y)) / fs

# Linear quantization
y_linear_quantized = np.round(y / (np.max(np.abs(y)) / (2 ** (N - 1)))) * (np.max(np.abs(y)) / (2 ** (N - 1)))

# Mu-law quantization

y_mu = np.sign(y)*np.log(1+u*abs(y))/np.log(1+u)
y_linear_mu_comp, _ = quant_bits(y_mu, N)
y_linear_mu = np.sign(y_linear_mu_comp)*(1/u)*((1+u)**(np.abs(y_linear_mu_comp))-1)

# play sounds
sd.play(y, fs)
sd.wait()
sd.play(y_linear_quantized, fs)
sd.wait()
sd.play(y_linear_mu, fs)

# Plot the signals
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(n, y, color='#7f7f7f')
plt.title('Original signal', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.grid(True)
plt.ylim(-1, 1)

plt.subplot(1, 3, 2)
plt.plot(n, y_linear_quantized, color='#1f77b4')
plt.title(f'{N}-bit Linear quantization', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.grid(True)
plt.ylim(-1, 1)

plt.subplot(1, 3, 3)
plt.plot(n, y_linear_mu, color='#d62728')
plt.title(f'{N}-bit \u03BC-law quantization (\u03BC={u})', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.grid(True)
plt.ylim(-1, 1)

plt.tight_layout()
plt.show()
