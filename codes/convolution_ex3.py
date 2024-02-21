"""
convolution_ex3.py
------------------

This script demonstrates the convolution of two sequences x(n) and
h(n) to obtain the output sequence y(n).
"""

import numpy as np

from tools.conv_m import conv_m

x = np.array([3, 11, 7, 0, -1, 4, 2])
nx = np.arange(-3, 4)

h = np.array([2, 3, 0, -5, 2, 1])
nh = np.arange(-1, 5)

[y,ny] = conv_m(x,nx,h,nh)

print('y(n) = ', y)
print('ny = ', ny)