"""
convolution_ex2.py
------------------

This script demonstrates the convolution of two sequences x(n) and
h(n) to obtain the output sequence y(n).
"""

import numpy as np
import matplotlib.pyplot as plt

from tools.conv_m import conv_m

# Input sequence x(n) and its indices nx
x = np.array([3, 11, 7, 0, -1, 4, 2])
nx = np.arange(-3, 4)

# Impulse response h(n) and its indices nh
h = np.array([2, 3, 0, -5, 2, 1])
nh = np.arange(-1, 5)

y, ny = conv_m(x, nx, h, nh)


fig = plt.figure(figsize=(8, 6))

# Plotting input sequence x(n)
ax1 = fig.add_subplot(2, 2, 1)
ax1.stem(nx, x)
ax1.set_title('x(k)')
ax1.set_xlabel('k')

# Plotting impulse response h(n)
ax2 = fig.add_subplot(2, 2, 2)
ax2.stem(nh, h)
ax2.set_title('h(k)')
ax2.set_xlabel('k')

# Plotting convolution result y(n)
ax3 = fig.add_subplot(2, 2,(3,4))
ax3.stem(ny, y)
ax3.set_title('Convolution y(k) = x(k) * h(k)')
ax3.set_xlabel('k')

plt.tight_layout()


fig = plt.figure(figsize=(8, 6))

# plot x(k) and h(k)
ax1 = fig.add_subplot(2, 2, 1)
ax1.stem(nx-0.05, x)
ax1.set(xlim=(-5,5), ylim=(-6,12))
ax1.stem(nh+0.05, h, ':r')
ax1.set_title('x(k) and h(k)')
ax1.set_xlabel('k')
ax1.text(-0.5, 11, 'solid: x    dashed: h')

# plot x(k) and h(-k)
ax2 = fig.add_subplot(2, 2, 2)
ax2.stem(nx-0.05, x)
ax2.set(xlim=(-5,5), ylim=(-6,12))
ax2.stem(-np.flip(nh)+0.05, np.flip(h), ':r')
ax2.set_title('x(k) and h(-k)')
ax2.set_xlabel('k')
ax2.text(-0.5, -1, 'n=0')
ax2.text(-0.5, 11, 'solid: x    dashed: h')

# plot x(k) and h(-1-k)
ax3 = fig.add_subplot(2, 2, 3)
ax3.stem(nx-0.05, x)
ax3.set(xlim=(-5,5), ylim=(-6,12))
ax3.stem(-np.flip(nh)+0.05-1, np.flip(h), ':r')
ax3.set_title('x(k) and h(-1-k)')
ax3.set_xlabel('k')
ax3.text(-1.5, -1, 'n=-1')
ax3.text(-0.5, 11, 'solid: x    dashed: h')

# plot x(k) and h(2-k)
ax4 = fig.add_subplot(2, 2, 4)
ax4.stem(nx-0.05, x)
ax4.set(xlim=(-5,5), ylim=(-6,12))
ax4.stem(-np.flip(nh)+0.05+2, np.flip(h), ':r')
ax4.set_title('x(k) and h(2-k)')
ax4.set_xlabel('k')
ax4.text(2-0.5, -1, 'n=2')
ax4.text(-0.5, 11, 'solid: x    dashed: h')

plt.tight_layout()
plt.show()
