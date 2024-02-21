"""
FIR_average_filter.py
----------------------

This script demonstrates the implementation of a moving average filter using
the lfilter function. The filter is applied to a given signal x(n) and the
filtered signal is plotted along with the original signal.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, lfilter


n = np.arange(0,41)
x = (1.02)**n + 1/2*np.cos(2*np.pi*n/8+np.pi/4)
baseline = (1.02)**n

filter_length = 7;  # filter length
# get filter transfer function
b = np.ones(filter_length)
# You then have to multiply the filter by the signal values, add them 
# all together and then divide the entire thing by the sum of the 
# coefficients.
a = np.sum(b)

# zero padding before filtering
x_padded = np.concatenate((np.zeros(len(b)), x, np.zeros(len(b))))
n_padded = np.concatenate((np.arange(-len(b),0), n, np.arange(n[-1]+1,len(b)+n[-1]+1)))

N = len(x_padded)

# Apply filtering using filtfilt function
filtered_x = lfilter(b, a, x_padded)

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
fig = plt.figure(figsize=(8, 8))

ax1 = fig.add_subplot(2, 1, 1)
ax1.stem(n_padded, x_padded, markerfmt='bo', linefmt='b-', basefmt='b-')
ax1.plot(n, baseline, 'r')
ax1.set_xlabel('Time (n)', fontsize=12)
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.set_xlim([n_padded[0], n_padded[-1]])
ax1.set_title('Original signal', fontsize=14)


ax2 = fig.add_subplot(2, 1, 2)
ax2.stem(n_padded, filtered_x, markerfmt='bo', linefmt='b-', basefmt='b-')
ax2.plot(n+np.floor(len(b)/2), baseline, 'r') # shift the baseline to the right by half the filter length
ax2.set_xlabel('Time (n)', fontsize=12)
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.set_xlim([n_padded[0], n_padded[-1]])
ax2.set_title(f'{filter_length}-point symmetric weighted moving average filter (lfilter)', fontsize=14)
left, bottom, width, height = (n[0], ax2.get_ylim()[0], len(b)-1, ax2.get_ylim()[1]-ax2.get_ylim()[0])
rect1 = plt.Rectangle((left, bottom), width, height,
                     facecolor="red", alpha=0.2)
ax2.add_patch(rect1)
left, bottom, width, height = (n[-1], ax2.get_ylim()[0], len(b)-1, ax2.get_ylim()[1]-ax2.get_ylim()[0])
rect2 = plt.Rectangle((left, bottom), width, height,
                     facecolor="red", alpha=0.2)
ax2.add_patch(rect2)
plt.tight_layout()

plt.show()