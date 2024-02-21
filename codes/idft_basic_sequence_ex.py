"""
idft_basic_sequence_ex.py
-------------------------

This script provides an example for using the coded IDFT function.
It compares the outputs and elapsed time between IDFT and IFFT
"""

from time import time
import numpy as np
import matplotlib.pyplot as plt

from tools.idft import idft, idft_for_loop

# generate a simple 9-point sequence
Xk = [1, 2, 3, 4, 5, 9, 8, 7, 6, 5]
N = 9 # Number of points for IDFT/IFFT

# IDFT computation
idft_time = time()
xn = idft(Xk, N)
# xn = idft_for_loop(Xk, N) # foor loop version
print(f'IDFT computation time: {time()-idft_time:2.5f} seconds')

fig = plt.figure(figsize=(8, 4))

ax1 = fig.add_subplot(1, 2, 1)
ax1.stem(np.arange(N), np.real(xn))
ax1.set_title('Recovered signal (IDFT)', fontsize=14)
ax1.set_xlabel('Time', fontsize=12)
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.grid(True)

# IFFT computation
ifft_time = time()
xn_ifft = np.fft.ifft(Xk, N)
print(f'IFFT computation time: {time()-ifft_time:2.5f} seconds')

ax2 = fig.add_subplot(1, 2, 2)
ax2.stem(np.arange(N), np.real(xn_ifft))
ax2.set_title('Recovered signal (IFFT)', fontsize=14) 
ax2.set_xlabel('Time', fontsize=12)
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.grid(True)

plt.show()