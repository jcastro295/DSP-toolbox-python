"""
idft_noisy_signal_ex.py
-----------------------

This script provides an example for using the coded IDFT function.
It compares the outputs and elapsed time between IDFT and IFFT
"""

from time import time
import numpy as np
import matplotlib.pyplot as plt

from tools.idft import idft, idft_for_loop

Fs = 1000            # Sampling frequency                    
T = 1/Fs             # Sampling period       
N = 500              # Length of signal - Number of points for DFT/FFT
t = np.arange(N)*T   # Time vector
f = Fs/N*np.arange(N)# Frequency vector

# generate a composite of noisy sine signals at 10 & 20 Hz
S = 0.8 + 0.7*np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
xn = S + 0.5*np.random.randn(len(t))

# get X(k) from x(n) using FFT
Xk = np.fft.fft(xn) 

# IDFT computation
idft_time = time()
xn_idft = idft(Xk, N)
# xn_idft = idft_for_loop(Xk, N) # foor loop version
print(f'IDFT computation time: {time()-idft_time:2.5f} seconds')

fig = plt.figure(figsize=(10, 4))

ax1 = fig.add_subplot(1, 3, 1)
ax1.plot(t, xn)
ax1.set_title('Original signal x(n)', fontsize=14)
ax1.set_xlabel('Time', fontsize=12)
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.set_xlim([t[0], t[-1]])
ax1.grid(True)

ax2 = fig.add_subplot(1, 3, 2)
ax2.plot(t, np.real(xn_idft))
ax2.set_title('Recovered signal (IDFT)', fontsize=14)
ax2.set_xlabel('Time', fontsize=12)
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.set_xlim([t[0], t[-1]])
ax2.grid(True)

# IFFT computation
ifft_time = time()
xn_ifft = np.fft.ifft(Xk, N)
print(f'IFFT computation time: {time()-ifft_time:2.5f} seconds')

ax3 = fig.add_subplot(1, 3, 3)
ax3.plot(t, np.real(xn_ifft))
ax3.set_title('Recovered signal (IFFT)', fontsize=14)
ax3.set_xlabel('Time', fontsize=12)
ax3.set_ylabel('Amplitude', fontsize=12)
ax3.set_xlim([t[0], t[-1]])
ax3.grid(True)

fig.tight_layout()

plt.show()