"""
IIR_filter_ex.py
-----------------

This script demonstrates the implementation of an IIR filter using the
difference equation:
    y(n) = 5*x(n) + 0.8*y(n-1)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from tools.diracseq import diracseq

n = np.arange(-3,8)
x = 2*diracseq(n,0) - 3*diracseq(n,1) + 2*diracseq(n,3)

# filter definition
# y(n) = 5*x(n) + 0.8*y(n-1)
b = [0, 5] # x(n-1), x(n)
a = [0.8, 0] # y(n-1) y(n)

N = len(x)

filtered_x = np.zeros(N)
# compute y(n) for n=0
filtered_x[0] = np.sum(b*x[:len(b)-1])

# Apply the filter
for i in range(N-(max(len(a),len(b)))):
    v_n = np.sum(b*x[i:i+len(a)])
    
    filtered_x[i+1] = v_n + np.sum(a*filtered_x[i:i+len(b)-1]); 

# Plot the original and filtered signals
fig = plt.figure(figsize=(8, 8))

ax1 = fig.add_subplot(3, 1, 1)
ax1.stem(n, x, markerfmt='bo', linefmt='b-', basefmt='b-')
ax1.set_xlabel('Time (n)', fontsize=12)
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.set_xlim([n[0], n[-1]])
ax1.set_title('Original signal', fontsize=14)

ax2 = fig.add_subplot(3, 1, 2)
ax2.stem(n, filtered_x, markerfmt='bo', linefmt='b-', basefmt='b-')
ax2.set_xlabel('Time (n)', fontsize=12)
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.set_xlim([n[0], n[-1]])
ax2.set_title('Filtered signal (hardcoded)', fontsize=14)

# using scipy lfilter function
h = lfilter(np.flip(b), np.insert(-np.flip(a[:-1]), 0, 1) , x)

ax3 = fig.add_subplot(3, 1, 3)
ax3.stem(n, h, markerfmt='bo', linefmt='b-', basefmt='b-')
ax3.set_xlabel('Time (n)', fontsize=12)
ax3.set_ylabel('Amplitude', fontsize=12)
ax3.set_xlim([n[0], n[-1]])
ax3.set_title('Filtered signal (filter)', fontsize=14)

plt.tight_layout()

plt.show()
