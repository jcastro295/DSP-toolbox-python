"""
even_odd_signal_synthesis.py
----------------------------

This script demonstrates the synthesis of a signal x(n) into its even and odd
parts x_e(n) and x_o(n) using the evenodd function.
"""

import numpy as np
import matplotlib.pyplot as plt

from tools.evenodd import evenodd
from tools.stepseq import stepseq


n = np.arange(0, 11)
x1, n1 = stepseq(0, 0, 10)
x2, n2 = stepseq(10, 0, 10)
x = x1 - x2

xe, xo, m = evenodd(x, n)

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.stem(n1, x1)
plt.stem(n2, x2, 'r--')
plt.title('x_1(n) = u(n) | x_2(n) = u(n-10)', fontsize=14)
plt.xlabel('n', fontsize=12)
plt.ylabel('x_1(n) | x_2(n)', fontsize=12)
plt.axis([-10, 10, 0, 1.2])

plt.subplot(2, 2, 2)
plt.stem(n, x)
plt.title('Rectangular pulse [x(n) = x_1(n) - x_2(n)]', fontsize=14)
plt.xlabel('n', fontsize=12)
plt.ylabel('x(n)', fontsize=12)
plt.axis([-10, 10, 0, 1.2])

plt.subplot(2, 2, 3)
plt.stem(m, xe)
plt.title('Even Part [x_e(n) = 0.5*(x(n) + x(-n))]', fontsize=14)
plt.xlabel('n', fontsize=12)
plt.ylabel('x_e(n)', fontsize=12)
plt.axis([-10, 10, 0, 1.2])

plt.subplot(2, 2, 4)
plt.stem(m, xo)
plt.title('Odd Part [x_o(n) = 0.5*(x(n) - x(-n))]', fontsize=14)
plt.xlabel('n', fontsize=12)
plt.ylabel('x_o(n)', fontsize=12)
plt.axis([-10, 10, -0.6, 0.6])

plt.tight_layout()
plt.show()
