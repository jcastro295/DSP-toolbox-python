"""
convolution_ex1.py
------------------

This script demonstrates the convolution of two sequences x(n) and 
h(n) to obtain the output sequence y(n).
"""

import numpy as np
import matplotlib.pyplot as plt

from tools.stepseq import stepseq


n = np.arange(-5, 51)
u1, _ = stepseq(0, -5, 50)
u2, _ = stepseq(10, -5, 50)

# input x(n)
x = u1 - u2

# impulse response h(n)
h = (0.9**n) * u1

# output response y(n)
y = (10 * (1 - (0.9) ** (n + 1))) * (u1 - u2) + (10 * (1 - (0.9) ** 10) * (0.9) ** (n - 9)) * u2

# Plotting
plt.figure(figsize=(8, 6))

plt.subplot(3, 1, 1)
plt.stem(n, x)
plt.title('Input Sequence x(n)')
plt.xlabel('n')
plt.ylabel('x(n)')
plt.xlim([-5, 50])

plt.subplot(3, 1, 2)
plt.stem(n, h)
plt.title('Impulse Response h(n)')
plt.xlabel('n')
plt.ylabel('h(n)')
plt.xlim([-5, 50])

plt.subplot(3, 1, 3)
plt.stem(n, y)
plt.title('Output Sequence y(n) = x(n)*h(n) (Convolution)')
plt.xlabel('n')
plt.ylabel('y(n)')
plt.xlim([-5, 50])

plt.tight_layout()
plt.show()
