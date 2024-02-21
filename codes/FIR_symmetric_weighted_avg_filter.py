"""
FIR_symmetric_weighted_avg_filter.py
-------------------------------------

This script demonstrates the implementation of a symmetric weighted moving average filter
using the filtfilt function. The filter is applied to a given ECG signal and the filtered
signal is plotted along with the original signal.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import freqz, filtfilt


data_folder = 'data'

# Loading ECG data: The first column is the time vector, 
# and the second column is the voltage vector of the signal.
data = pd.read_excel(os.path.join(data_folder, 'ECG_data.xlsx'))

time = data['time'].values[1:]
voltage = data['Amplitude'].values[1:]

n = 9  # filter length. It has to be and odd number
n = n - ((n+1) % 2)  # make sure that n is odd

# Pick only the first 150 points in the data
time = time[:150]
voltage = voltage[:150]

# get filter transfer function
# Adds weights of one to the first and last points in the filter, and 
# increases the weight value by 1 as you move closer to the center point. 
# Thus a n=5 point filter would have weights of [1,2,3,2,1], respectively.
b = np.convolve(np.ones(np.round(n/2).astype(int) + 1), 
                np.ones(np.round(n/2).astype(int) + 1))
# we need to divide the entire thing by the sum of the 
# coefficients, which for a five point filter, it would be 1+2+3+2+1 = 9.
a = np.sum(b)

# Apply filtering using filtfilt function
filtered_voltage = filtfilt(b, a, voltage)

# we can analize the magnitude and phase of the filter's impulse response
fig = plt.figure(figsize=(8, 8))
frequencies, response = freqz(b, a, worN=2**12)

ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(frequencies/(2*np.pi), 20*np.log10(np.abs(response)))
ax1.set_xlabel('Frequency (x pi rad/sample)', fontsize=12)
ax1.set_ylabel('Magnitude (dB)', fontsize=12)
ax1.set_title('Magnitude', fontsize=14)
ax1.set_xlim([frequencies[0]/(2*np.pi), frequencies[-1]/(2*np.pi)])
ax1.grid(True)

ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(frequencies/(2*np.pi), np.angle(response, deg=True))
ax2.set_xlabel('Frequency (x pi rad/sample)', fontsize=12)
ax2.set_ylabel('Phase (degrees)', fontsize=12)
ax2.set_title('Phase', fontsize=14)
ax2.set_xlim([frequencies[0]/(2*np.pi), frequencies[-1]/(2*np.pi)])
ax2.grid(True)

plt.tight_layout()

# Plot the original and filtered signals
plt.figure(figsize=(8, 6))
plt.stem(time, voltage, markerfmt='bo', linefmt='b-', basefmt='b-')
plt.stem(time, filtered_voltage, markerfmt='ro', linefmt='r-', basefmt='r-')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Voltage', fontsize=12)
plt.xlim([time[0], time[-1]])
plt.title(f'{n}-point symmetric weighted moving average filter (hardcoded)', fontsize=14)
plt.legend(['Original signal', 'Filtered signal'], loc='upper right', fontsize=12)
plt.show()