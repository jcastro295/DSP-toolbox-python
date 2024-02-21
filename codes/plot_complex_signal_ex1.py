"""
plot_complex_signal_ex1.py
--------------------------

This script demonstrates how to plot the real, imaginary, magnitude, and phase
parts of a complex signal x(n) = exp((-0.1+j0.3)n), -10 <= n <= 10.
"""

import numpy as np
import matplotlib.pyplot as plt

# a) x(n) = exp((-0.1+j0.3)n), -10 <= n <= 10;

n = np.arange(-10, 11, 1)
alpha = -0.1 + 0.3j
x = np.exp(alpha * n)

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.stem(n, np.real(x))
plt.title('real part')
plt.xlabel('n')

plt.subplot(2, 2, 2)
plt.stem(n, np.imag(x))
plt.title('imaginary part')
plt.xlabel('n')

plt.subplot(2, 2, 3)
plt.stem(n, np.abs(x))
plt.title('magnitude part')
plt.xlabel('n')

plt.subplot(2, 2, 4)
plt.stem(n, np.angle(x, deg=True))
plt.title('phase part')
plt.xlabel('n')

plt.tight_layout()
plt.show()
