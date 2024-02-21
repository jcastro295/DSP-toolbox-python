"""
linear_vs_circular_conv.py
---------------------------

This script computes and compare the linear convolution
and circular convolution
"""

import numpy as np
import matplotlib.pyplot as plt


# For two vectors, x and y, the circular convolution is equal to the 
# inverse discrete Fourier transform (DFT) of the product of the vectors' 
# DFTs. Knowing the conditions under which linear and circular convolution 
# are equivalent allows you to use the DFT to efficiently compute linear 
# convolutions. The linear convolution of an N-point vector, x, and an 
# L-point vector, y, has length N + L - 1.

# Create two vectors, x and y, and compute the linear 
# convolution of the two vectors.
x = [2, 1, 2, 1]
y = [1, 2, 3]
clin = np.convolve(x,y) # The output has length 4+3-1.

print("Linear convolution of x and y: \n", clin, "\n")

# The circular convolution is the IDFT of the product of two DFT.
# Before computing, we have to make sure both vectors are the same size
max_length = np.max([len(x), len(y)])
if len(x) < len(y):
    xp = np.concatenate([x, np.zeros(max_length-len(x))])
    yp = y.copy()
else:
    xp = x.copy()
    yp = np.concatenate([y, np.zeros(max_length-len(y))])

ccirc = np.real(np.fft.ifft(np.fft.fft(xp)*np.fft.fft(yp)))

print("Circular convolution of x and y: \n", ccirc, "\n")

# Notice that due to the cyclic nature of DFT, the convolution operator 
# is not equal to the one we compute in continous time domain, as shown
# below. In this contex, both convolution are not equivalent.
fig = plt.figure(figsize=(9, 8))

ax1 = fig.add_subplot(2,2,1)
ax1.stem(np.arange(len(xp)), xp)
ax1.set_title('x(n)', fontsize=14)
ax1.set_xlabel('Time', fontsize=12)
ax1.set_ylabel('Amplitude', fontsize=12)

ax2 = fig.add_subplot(2,2,2)
ax2.stem(np.arange(len(yp)), yp)
ax2.set_title('y(n)', fontsize=14)
ax2.set_xlabel('Time', fontsize=12)
ax2.set_ylabel('Amplitude', fontsize=12)

ax3 = fig.add_subplot(2,2,3)
ax3.stem(np.arange(len(clin)), clin)
ax3.set_title('Linear Convolution of x and y', fontsize=14)
ax3.set_xlabel('Time', fontsize=12)
ax3.set_ylabel('Amplitude', fontsize=12)

ax4 = fig.add_subplot(2,2,4)
ax4.stem(np.arange(len(ccirc)), ccirc)
ax4.set_title('Circular Convolution of x and y', fontsize=14)
ax4.set_xlabel('Time', fontsize=12)
ax4.set_ylabel('Amplitude', fontsize=12)

fig.tight_layout()

N = len(x) + len(y) - 1

# Pad both vectors with zeros to length 4+3-1. Obtain the DFT of both 
# vectors, multiply the DFTs, and obtain the inverse DFT of the product.
xpad = np.concatenate([x, np.zeros(N-len(x))])
ypad = np.concatenate([y, np.zeros(N-len(y))])
ccirc = np.real(np.fft.ifft(np.fft.fft(xpad)*np.fft.fft(ypad)))

print("Circular convolution of x padded and y padded : \n", ccirc, "\n")

# The circular convolution of the zero-padded vectors, xpad and ypad, is 
# equivalent to the linear convolution of x and y. You retain all the 
# elements of ccirc because the output has length 4+3-1.

# Plot the output of linear convolution and the inverse of the DFT product 
# to show the equivalence.

fig = plt.figure(figsize=(9, 8))

ax1 = fig.add_subplot(2,2,1)
ax1.stem(np.arange(len(xpad)), xpad)
ax1.set_title('x(n) (zero padding)', fontsize=14)
ax1.set_xlabel('Time', fontsize=12)
ax1.set_ylabel('Amplitude', fontsize=12)

ax2 = fig.add_subplot(2,2,2)
ax2.stem(np.arange(len(ypad)), ypad)
ax2.set_title('y(n) (zero padding)', fontsize=14)
ax2.set_xlabel('Time', fontsize=12)
ax2.set_ylabel('Amplitude', fontsize=12)

ax3 = fig.add_subplot(2,2,3)
ax3.stem(np.arange(N), clin)
ax3.set_title('Linear Convolution of x and y', fontsize=14)
ax3.set_xlabel('Time', fontsize=12)
ax3.set_ylabel('Amplitude', fontsize=12)

ax4 = fig.add_subplot(2,2,4)
ax4.stem(np.arange(N), ccirc)
ax4.set_title('Circular Convolution of xpad and ypad', fontsize=14)
ax4.set_xlabel('Time', fontsize=12)
ax4.set_ylabel('Amplitude', fontsize=12)

fig.tight_layout()

# We can pad as most any zeros we want, and retain only the 4+3-1 elements
# of the computed circular convolution.
# Pad the vectors to length 12 and obtain the circular convolution using 
# the inverse DFT of the product of the DFTs. Retain only the first 4+3-1 
# elements to produce an equivalent result to linear convolution.
N = len(x) + len(y) - 1
xpad = np.concatenate([x, np.zeros(12-len(x))])
ypad = np.concatenate([y, np.zeros(12-len(y))])
ccirc = np.real(np.fft.ifft(np.fft.fft(xpad)*np.fft.fft(ypad)))
ccirc = ccirc[:N]

print("Circular convolution of x padded and y padded : \n", ccirc, "\n")

plt.show()