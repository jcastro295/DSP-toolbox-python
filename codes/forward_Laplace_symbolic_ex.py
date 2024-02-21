"""
forward_Laplace_symbolic_ex.py
-------------------------------

This script demonstrates the computation of the Laplace transform of some
basic signals using the sympy library.
"""

from sympy import (laplace_transform, symbols, sqrt, diff, pretty, 
                   exp, DiracDelta, Heaviside, Function)

# Define symbolic variables
a, t, s, x, y = symbols('a t s x y', positive=True)

# Ex 1. Compute the Laplace transform of 1/sqrt(x).
f1 = 1/sqrt(x)
F1 = laplace_transform(f1, x, s)
print("\nEx1. 1/sqrt(x)")
print(pretty(F1))

# Ex 2. Compute the Laplace transform of exp(-a*t).
f2 = exp(-a*t)
F2 = laplace_transform(f2, t, s)
print("\nEx2. exp(-a*t)")
print(pretty(F2))

# Ex 3. Compute the Laplace transforms of the Dirac and Heaviside functions.
f3_dirac = DiracDelta(t - a)
F3_dirac = laplace_transform(f3_dirac, t, s)
print("\nEx3.1. Dirac function")
print(pretty(F3_dirac))

f3_heaviside = Heaviside(t - a)
F3_heaviside = laplace_transform(f3_heaviside, t, s)
print("\nEx3.2. Heaviside function")
print(pretty(F3_heaviside))

# Ex 4. Show that the Laplace transform of the derivative of a function.
f4 = Function('f')(t)
Df4 = diff(f4, t)
F4 = laplace_transform(Df4, t, s)
print("\nEx4. Laplace transform of the derivative")
print(pretty(F4))
