"""
fft_comparison.py
------------------

This script provides an example for using the coded DFT function.
It compares the outputs and elapsed time between DFT and FFT
"""

from time import time
import numpy as np
import matplotlib.pyplot as plt

from tools.dft import dft, dft_for_loop
from tools.fft import fft_coded

# generate a simple sequence
# for this fft implementation, xn must be a power of 2
xn = np.random.rand(2**10)
N = len(xn) # Number of points for DFT/FFT

# DFT computation
dft_time = time()
Xk_dft = dft(xn, N)
# Xk_dft = dft_for_loop(Xk, N) # foor loop version
dft_end_time = time()-dft_time
print(f'DFT computation time: {dft_end_time:2.5f} seconds')

fig = plt.figure(figsize=(14, 8))

ax1 = fig.add_subplot(2, 3, 1)
_, line1, _ = ax1.stem(np.arange(N), np.abs(Xk_dft))
ax1.set_title('Magnitude of the DFT sequence', fontsize=14) 
ax1.set_xlabel('Frequency', fontsize=12) 
ax1.set_ylabel('Magnitude', fontsize=12)
ax1.grid(True)

ax2 = fig.add_subplot(2, 3, 4) 
ax2.stem(np.arange(N), np.angle(Xk_dft))
ax2.set_title('Phase of the DFT sequence', fontsize=14)
ax2.set_xlabel('Frequency', fontsize=12)
ax2.set_ylabel('Phase', fontsize=12)
ax2.grid(True)

# FFT computation (using the coded fft function)
fft_time = time() 
Xk_fft = fft_coded(xn)
fft_end_time = time()-fft_time
print(f'FFT computation time: {fft_end_time:2.5f} seconds')

ax3 = fig.add_subplot(2, 3, 2)
_, line2, _ = ax3.stem(np.arange(N), np.abs(Xk_fft))
ax3.set_title('Magnitude of the FFT sequence', fontsize=14) 
ax3.set_xlabel('Frequency', fontsize=12) 
ax3.set_ylabel('Magnitude', fontsize=12)
ax3.grid(True)

ax4 = fig.add_subplot(2, 3, 5) 
ax4.stem(np.arange(N), np.angle(Xk_fft))
ax4.set_title('Phase of the FFT sequence', fontsize=14)
ax4.set_xlabel('Frequency', fontsize=12)
ax4.set_ylabel('Phase', fontsize=12)
ax4.grid(True)

# FFT computation (using the numpy fft function)
fft_time_numpy = time() 
Xk_fft_numpy = np.fft.fft(xn, N)
fft_end_time_numpy = time()-fft_time_numpy
print(f'FFT computation time (numpy): {fft_end_time_numpy:2.5f} seconds')

ax5 = fig.add_subplot(2, 3, 3)
_, line3, _ = ax5.stem(np.arange(N), np.abs(Xk_fft_numpy))
ax5.set_title('Magnitude of the FFT sequence (numpy)', fontsize=14) 
ax5.set_xlabel('Frequency', fontsize=12) 
ax5.set_ylabel('Magnitude', fontsize=12)
ax5.grid(True)

ax6 = fig.add_subplot(2, 3, 6) 
ax6.stem(np.arange(N), np.angle(Xk_fft_numpy))
ax6.set_title('Phase of the FFT sequence (numpy)', fontsize=14)
ax6.set_xlabel('Frequency', fontsize=12)
ax6.set_ylabel('Phase', fontsize=12)
ax6.grid(True)

fig.tight_layout()

ax1.legend([line1], [f'Time: {dft_end_time:2.5f}'], loc='upper right')
ax3.legend([line2], [f'Time: {fft_end_time:2.5f} \n (x{int(dft_end_time/fft_end_time)} faster than DFT)'], 
          loc='upper right')
ax5.legend([line3], [f'Time: {fft_end_time_numpy:2.5f}\n (x{int(dft_end_time/fft_end_time_numpy)} faster than DFT)'],
          loc='upper right')

plt.show()
