"""
quantization_audio_ex.py
------------------------

This script demonstrates the quantization of an audio signal using linear
quantization.
"""

import os
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

from tools.quant_bits import quant_bits


data_folder = 'data'

# Parameters
N = 32  # bit-depth

# Read the audio file
fs, y = wavfile.read(os.path.join(data_folder, 'flute.wav'))
y = y / 2**np.ceil(np.log2(y.max())) # Normalize to [-1, 1]

# Crop the signal
y = y[:int(0.23 * len(y))]

n = np.arange(0, len(y)) / fs

# Linear quantization
y_linear_quantized, _ = quant_bits(y, N)

# play sounds
sd.play(y, fs)
sd.wait()
sd.play(y_linear_quantized, fs)

# Plot the signals
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(n, y, color='#7f7f7f')
plt.title('Original signal', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(n, y_linear_quantized, color='#1f77b4')
plt.title(f'{N}-bit Linear quantization', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.show()
