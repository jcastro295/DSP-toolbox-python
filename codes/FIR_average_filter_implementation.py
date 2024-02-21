"""
FIR_average_filter_implementation.py
------------------------------------

This script demonstrates the implementation of a moving average filter using
a for loop and the numpy mean function. The filter is applied to a given signal
x(n) and the filtered signal is plotted along with the original signal.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz


n = np.arange(0,41)
x = (1.02)**n + 1/2*np.cos(2*np.pi*n/8+np.pi/4)
baseline = (1.02)**n

filter_length = 7;  # filter length
kernel = np.ones(filter_length)

# zero padding before filtering
x_padded = np.concatenate((np.zeros(len(kernel)), x, np.zeros(len(kernel))))
n_padded = np.concatenate((np.arange(-len(kernel),0), n, np.arange(n[-1]+1,len(kernel)+n[-1]+1)))

N = len(x_padded)

filtered_x = x_padded.copy()

# Apply the moving average filter
for i in range(N - len(kernel)):
    # You then have to multiply the filter by the signal values, add them 
    # all together and then divide the entire thing by the sum of the 
    # coefficients.
    filtered_x[i+len(kernel)-1] = np.mean(x_padded[i : i+len(kernel)]*kernel)


# we can analize the magnitude and phase of the filter's impulse response
fig = plt.figure(figsize=(8, 8))
frequencies, response = freqz(kernel, sum(kernel), worN=2**12)

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
ax2.stem(n_padded, filtered_x, markerfmt='bo', linefmt='b-', basefmt='b-') # shift the filtered signal to the right by the filter length
ax2.plot(n+np.floor(len(kernel)/2), baseline, 'r') # shift the baseline to the right by half the filter length
ax2.set_xlabel('Time (n)', fontsize=12)
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.set_xlim([n_padded[0], n_padded[-1]])
ax2.set_title(f'{filter_length}-point symmetric weighted moving average filter (hardcoded)', fontsize=14)
left, bottom, width, height = (n[0], ax2.get_ylim()[0], len(kernel)-1, ax2.get_ylim()[1]-ax2.get_ylim()[0])
rect1 = plt.Rectangle((left, bottom), width, height,
                     facecolor="red", alpha=0.2)
ax2.add_patch(rect1)
left, bottom, width, height = (n[-1], ax2.get_ylim()[0], len(kernel)-1, ax2.get_ylim()[1]-ax2.get_ylim()[0])
rect2 = plt.Rectangle((left, bottom), width, height,
                     facecolor="red", alpha=0.2)
ax2.add_patch(rect2)

plt.tight_layout()

plt.show()
