"""
difference_equation_ex1.py
---------------------------

This script demonstrates the calculation of impulse response and step response
of a discrete-time system given its difference equation coefficients. The script
also checks the stability of the system.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

from tools.impseq import impseq
from tools.stepseq import stepseq

# Given difference equation coefficients
a = [1, -1, 0.9]
b = [1]

# Part a) Calculate and plot the impulse response h(n) at n = -20,..., 100.
x, _ = impseq(0, -20, 120)
n = np.arange(-20, 121)

fig, ax = plt.subplots(2, 1, figsize=(8, 6))

h = lfilter(b, a, x)

ax[0].stem(n, h)
ax[0].set_title('Impulse Response')
ax[0].set_xlabel('n')
ax[0].set_ylabel('h(n)')
ax[0].set(xlim=(-20,120), ylim=(-1.1,1.1))

# Part b) Calculate and plot the unit step response s(n) at n = -20,..., 100.
x, _ = stepseq(0, -20, 120)
s = lfilter(b, a, x)
ax[1].stem(n, s)
ax[1].set_title('Step Response')
ax[1].set_xlabel('n')
ax[1].set_ylabel('s(n)')
ax[1].set(xlim=(-20,120), ylim=(-0.5,2.5))

plt.tight_layout()


# Part c) Check stability
roots = np.roots(a)
magnitude_roots = np.abs(roots)
print("Roots:", roots)
print("Magnitude of Roots:", magnitude_roots)

if all(magnitude_roots < 1):
    print("The system is stable.")
else:
    print("The system is not stable.")

plt.show()