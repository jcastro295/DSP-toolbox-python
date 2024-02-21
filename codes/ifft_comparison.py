"""
fft_comparison.py
------------------

This script provides an example for using the coded IDFT function.
It compares the outputs and elapsed time between IDFT and IFFT
"""

from time import time
import numpy as np
import matplotlib.pyplot as plt

from tools.idft import idft, idft_for_loop
from tools.ifft import ifft_coded

# generate a simple sequence
# for this fft implementation, Xk must be a power of 2
Xk = np.random.rand(2**10)
N = len(Xk) # Number of points for IDFT/IFFT

# IDFT computation
idft_time = time()
xn = idft(Xk, N)
# xn = idft_for_loop(Xk, N) # foor loop version
idft_end_time = time()-idft_time
print(f'IDFT computation time: {idft_end_time:2.5f} seconds')

fig = plt.figure(figsize=(13, 4))

ax1 = fig.add_subplot(1, 3, 1)
_, line1, _ = ax1.stem(np.arange(N), np.real(xn))
ax1.set_title('Recovered signal (IDFT)', fontsize=14)
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.set_xlabel('Time', fontsize=12)
ax1.grid(True)

# IFFT computation (using the coded ifft function)
ifft_time = time()
xn_ifft = ifft_coded(Xk)
ifft_end_time = time()-ifft_time
print(f'IFFT computation time: {ifft_end_time:2.5f} seconds')

ax2 = fig.add_subplot(1, 3, 2)
_, line2, _ = ax2.stem(np.arange(N), np.real(xn_ifft))
ax2.set_title('Recovered signal (IFFT)', fontsize=14) 
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.set_xlabel('Time', fontsize=12)
ax2.grid(True)

# IFFT computation (using the numpy ifft function)
ifft_time_numpy = time()
xn_ifft_numpy = np.fft.ifft(Xk, N)
ifft_end_time_numpy = time()-ifft_time_numpy
print(f'IFFT computation time: {ifft_end_time_numpy:2.5f} seconds')

ax3 = fig.add_subplot(1, 3, 3)
_, line3, _ = ax3.stem(np.arange(N), np.real(xn_ifft_numpy))
ax3.set_title('Recovered signal (IFFT) (numpy)', fontsize=14) 
ax3.set_ylabel('Amplitude', fontsize=12)
ax3.set_xlabel('Time', fontsize=12)
ax3.grid(True)

ax1.legend([line1], [f'Time: {idft_end_time:2.5f}'], loc='upper right')
ax2.legend([line2], [f'Time: {ifft_end_time:2.5f} \n (x{int(idft_end_time/ifft_end_time)} faster than DFT)'], 
          loc='upper right')
ax3.legend([line3], [f'Time: {ifft_end_time_numpy:2.5f}\n (x{int(idft_end_time/ifft_end_time_numpy)} faster than DFT)'],
          loc='upper right')

plt.show()