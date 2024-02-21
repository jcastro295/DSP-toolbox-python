"""
inverse_Fourier_symbolic_ex.py
-------------------------------

This script demonstrates the computation of the inverse Fourier transform of
some basic signals using the sympy library.
"""

from sympy import (inverse_fourier_transform, symbols, pretty,
                   Abs, exp, DiracDelta, Heaviside)

# Define symbolic variables
a, omega, t = symbols('a omega t')

# Ex 1. Compute the inverse Fourier transform of exp(-w^2/4).
F1 = exp(-omega**2/4)
f1 = inverse_fourier_transform(F1, omega, t)
print("\nEx1. exp(-w^2/4)")
print(pretty(f1))

# Ex 2. Compute the inverse Fourier transform of exp(-w^2-a^2).
F2 = exp(-omega**2 - a**2)
f2 = inverse_fourier_transform(F2, omega, t)
print("\nEx2. exp(-w^2-a^2)")
print(pretty(f2))

# Ex 3. Compute the inverse Fourier transform of expressions in terms of Dirac and Heaviside functions.
f3_1 = inverse_fourier_transform(DiracDelta(t), omega, t)
print("\nEx3.1. Dirac")
print(pretty(f3_1))

F3_2 = 2*exp(-Abs(omega)) - 1
f3_2 = inverse_fourier_transform(F3_2, omega, t)
print("\nEx3.2. Exponential & Dirac")
print(pretty(f3_2))

F3_3 = exp(-omega) * Heaviside(t)
f3_3 = inverse_fourier_transform(F3_3, omega, t)
print("\nEx3.3. Exponential and Heaviside")
print(pretty(f3_3))
