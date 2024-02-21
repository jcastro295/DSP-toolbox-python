"""
dft_noisy_signal_ex.py
-----------------------

This script provides an example for using the coded DFT function.
It compares the outputs and elapsed time between DFT and FFT
"""

from time import time
import numpy as np
import matplotlib.pyplot as plt

from tools.dft import dft, dft_for_loop

Fs = 1000             # Sampling frequency                    
T = 1/Fs              # Sampling period       
N = 1500              # Length of signal - Number of points for DFT/FFT
t = np.arange(N)*T    # Time vector
f = Fs/N*np.arange(N) # Frequency vector

# generate a composite of noisy sine signals at 50 & 120 Hz
S = 0.8 + 0.7*np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t)
xn = S + 2*np.random.randn(len(t))

# DFT computation
dft_time = time()
Xk_dft = dft(xn, N)
# Xk_dft = dft_for_loop(Xk, N) # foor loop version
print(f'DFT computation time: {time()-dft_time:2.5f} seconds')

fig = plt.figure(figsize=(8, 8))

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(f, np.abs(Xk_dft))
ax1.set_title('Magnitude of the DFT sequence', fontsize=14) 
ax1.set_xlabel('Frequency', fontsize=12) 
ax1.set_ylabel('Magnitude', fontsize=12)
ax1.set_xlim([f[0], f[-1]])
ax1.grid(True)

ax2 = fig.add_subplot(2, 2, 3) 
ax2.plot(f, np.angle(Xk_dft))
ax2.set_title('Phase of the DFT sequence', fontsize=14)
ax2.set_xlabel('Frequency', fontsize=12)
ax2.set_ylabel('Phase', fontsize=12)
ax2.set_xlim([f[0], f[-1]])
ax2.grid(True)

# FFT computation
fft_time = time() 
Xk_fft = np.fft.fft(xn, N)
print(f'FFT computation time: {time()-fft_time:2.5f} seconds')

ax3 = fig.add_subplot(2, 2, 2)
ax3.plot(f, np.abs(Xk_fft))
ax3.set_title('Magnitude of the FFT sequence', fontsize=14) 
ax3.set_xlabel('Frequency', fontsize=12) 
ax3.set_ylabel('Magnitude', fontsize=12)
ax3.set_xlim([t[0], f[-1]])
ax3.grid(True)

ax4 = fig.add_subplot(2, 2, 4) 
ax4.plot(f, np.angle(Xk_fft))
ax4.set_title('Phase of the FFT sequence', fontsize=14)
ax4.set_xlabel('Frequency', fontsize=12)
ax4.set_ylabel('Phase', fontsize=12)
ax4.set_xlim([f[0], f[-1]])
ax4.grid(True)

fig.tight_layout()

plt.show()