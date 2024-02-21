"""
inverse_Laplace_symbolic_ex.py
-------------------------------

This script demonstrates the computation of the inverse Laplace transform of
some basic signals using the sympy library.
"""

from sympy import (laplace_transform, inverse_laplace_transform, symbols, pretty,
                   integrate, exp, Heaviside)

# Define symbolic variables
a, s, t, tau = symbols('a s t tau', positive=True)

# Ex 1. Compute the inverse Laplace transform of 1/s^2.
F1 = 1/s**2
f1 = inverse_laplace_transform(F1, s, t)
print("\nEx1. 1/s^2")
print(pretty(f1))

# Ex 2. Compute the inverse Laplace transform of 1/(s-a)^2.
F2 = 1/(s-a)**2
f2 = inverse_laplace_transform(F2, s, t)
print("\nEx2. 1/(s-a)^2")
print(pretty(f2))

# Ex 3. Compute the following inverse Laplace transforms that involve the Dirac and Heaviside functions.
f3_1 = inverse_laplace_transform(1, s, t)
print("\nEx3.1. Dirac")
print(pretty(f3_1))

F3_2 = exp(-2*s)/(s**2 + 1)
f3_2 = inverse_laplace_transform(F3_2, s, t)
print("\nEx3.2. Heaviside")
print(pretty(f3_2))

# Ex 4. Create two functions f(t)=heaviside(t) and g(t)=exp(−t). Find the 
# Laplace transforms of the two functions by using laplace.
# Because the Laplace transform is defined as a unilateral or one-sided 
# transform, it only applies to the signals in the region t≥0.
def f(t):
    return Heaviside(t)

def g(t):
    return exp(-t)

F = laplace_transform(f(t), t, s)
G = laplace_transform(g(t), t, s)

h = inverse_laplace_transform(F[0]*G[0], s, t)
print("\nEx4. h(t)")
print(pretty(h))

# According to the convolution theorem for causal signals, the inverse 
# Laplace transform of this product is equal to the convolution.
conv_fg = integrate(f(tau)*g(t-tau), (tau, 0, t))
print('\nEx4. conv_fg')
print(pretty(conv_fg))