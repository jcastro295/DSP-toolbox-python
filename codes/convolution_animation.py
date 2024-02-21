"""
convolution_animation.py
------------------------

This code is an animation of the convolution of two discrete-time signals
x(n) and h(n). The code uses numpy to calculate the convolution of the two 
signals and then animates the process of convolution. The code also compares 
the result with the numpy's convolve function.
"""

import numpy as np
import matplotlib.pyplot as plt

step_size = 0.1
nt_x = np.arange(0, 1 + step_size, step_size)
nt_h = np.arange(0, 2 + step_size, step_size)

# x = np.array([0, 1, 2, 1, 1, 1, 4, 3, 4, 3, 2, 1, 0])    # x(n) random signal
# x = np.array([1, 1, 1, 1, 1, 1, 1, 1])                   # square
x = np.exp(nt_x);                                        # exponential
# x = np.array([1, 2, 3, 4, 5, 6, 7, 8])                   # ramp
# x = np.concatenate(([1], np.zeros(10)))                  # impulse

# h = np.array([0, 1, 2, 1, 1, 1, 4, 3, 4, 3, 2, 1, 0])    # h(n) random signal
h = np.array([1, 1, 1, 1, 1, 1, 1, 1])                   # h(n) square
# h = np.exp(nt_h)                                         # exponential
# h = nt_h                                                 # ramp
# h = np.concatenate(([1], np.zeros(10)))                  # impulse

m = len(x)
n = len(h)

k = 2 * max(len(x), len(h))

hi = np.flip(h)
X = np.concatenate((x, np.zeros(2 * k - m)))
X = np.roll(X, -k)
H = np.concatenate((h, np.zeros(2 * k - n)))
H = np.roll(H, -k)
xn = np.arange(-k, k) * step_size
Y = np.zeros(2 * k)
p = np.zeros(2 * k)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6))

ax1.stem(xn, X, linefmt='g', markerfmt='go')
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.set_title('x(n)', fontsize=12)

stem2 = ax2.stem(xn, H, linefmt='b', markerfmt='bo')
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.set_title('h(n)', fontsize=12)

stem3 = ax3.stem(xn, Y, linefmt='r', markerfmt='ro')
ax3.set_xlabel('Time (n)', fontsize=12)
ax3.set_ylabel('Amplitude', fontsize=12)
ax3.set_title('y(n) = x(n)*h(n)', fontsize=14)

Hi = np.concatenate((hi, np.zeros(2*k-n)))
plt.tight_layout()

# Animation loop
for i in range(2*k-n):

    p = X*Hi
    Y[i+n-1] = np.sum(p)

    ax2.cla()
    ax2.stem(xn, Hi, linefmt='b', markerfmt='bo')
    ax2.set_ylabel('Amplitude', fontsize=12)
    ax2.set_title('h(n)', fontsize=12)
    ax2.set_ylim(0, np.max(H)+1)
    
    ax3.cla()
    ax3.stem(xn, Y, linefmt='r', markerfmt='ro')
    ax3.set_xlabel('Time (n)', fontsize=12)
    ax3.set_ylabel('Amplitude', fontsize=12)
    ax3.set_title('y(n) = x(n)*h(n)', fontsize=14)
    ax3.set_ylim(0, np.max(np.convolve(x,h))+1)

    Hi = np.roll(Hi, 1)
    if i == 0:
        plt.pause(1)
    else:
        plt.pause(0.1)


# compare with matlab
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

Y_conv = np.convolve(x, h)
n_matlab = np.arange(-k, k)
Y_matlab = np.zeros(2*k); # zero padding
Y_matlab[int(np.round(len(n_matlab)/2,)):int(np.round(len(n_matlab)/2)+len(Y_conv))] = Y_conv

ax1.stem(xn, Y, linefmt='b', markerfmt='bo')
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.set_xlabel('x(n)', fontsize=12)
ax1.set_title('y(n) calculated using this code', fontsize=14)
ax1.set_ylim(0, np.max(Y_matlab)+1)

ax2.stem(n_matlab, Y_matlab, linefmt='r', markerfmt='ro')
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.set_xlabel('x(n)', fontsize=12)
ax2.set_title('y(n) calculated using numpy', fontsize=14)
ax2.set_ylim(0, np.max(Y_matlab)+1)

plt.tight_layout()

plt.show()
