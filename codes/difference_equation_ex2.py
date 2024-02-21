"""
difference_equation_ex2.py
---------------------------

This script demonstrates the calculation of the output of a discrete-time system
given its difference equation coefficients using for loop and filter function.
"""

import numpy as np
from scipy.signal import lfilter

# Example 1
x = [1, -2, 3, -4]
n_terms = len(x)
y = np.zeros(n_terms + 1)
y[0] = 1
y[1] = 1

for k in range(1, n_terms):
    y[k + 1] = -2 * y[k] + 3 * y[k - 1] + 1.5 * x[k] + 4 * x[k - 1]

print("Ex1. For loop solution, y(5) =", y[4])
print("Ex1. Result array:", y)

# Example 2
n_terms = 100
y = np.zeros(n_terms)
y[0] = 2
y[1] = 0

for k in range(2, n_terms):
    y[k] = (1 + 6 * y[k - 1] - 2 * y[k - 2]) / 8

print("Ex2. For loop solution, y(97:100)")
print(y[-4:])

# Example 3
n_terms = 50
y = np.zeros(n_terms)
y[0] = 1
y[1] = 1

for i in range(1, n_terms - 1):
    y[i + 1] = 5 / 2 * y[i] + y[i - 1]

print("Ex3. For loop solution, y(47:50)")
print(y[-4:])

# Using filter function
a = [1, -5/2, -1]
b = [0]
ic = [1, 1]

y, _ = lfilter(b, a, np.ones(n_terms-1), zi=ic)
y_filter = np.concatenate(([ic[0]], y))
print("Ex3. Filter function solution, y(47:50)")
print(y_filter[-4:])
