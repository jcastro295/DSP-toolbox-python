"""
forward_Fourier_symbolic_ex.py
-------------------------------

This script demonstrates the computation of the Fourier transform of some
basic signals using the sympy library.
"""

from sympy import (fourier_transform, symbols, pretty, 
                   Abs, cos, exp, sign, Heaviside, DiracDelta)

# Define symbolic variables
a, b, t, omega = symbols('a b t omega')

# Ex 1. Rectangular pulse.
f_rect = Heaviside(t + 1/2) - Heaviside(t - 1/2)
F_rect = fourier_transform(f_rect, t, omega)
print("\nEx1. Rectangular pulse")
print(pretty(F_rect))

# Ex 2. Unit impulse (Dirac delta).
f_dirac = DiracDelta(t)
F_dirac = fourier_transform(f_dirac, t, omega)
print("\nEx2. Unit impulse (Dirac delta)")
print(pretty(F_dirac))

# Ex 3. Absolute value.
f_abs = a * Abs(t)
F_abs = fourier_transform(f_abs, t, omega)
print("\nEx3. Absolute value")
print(pretty(F_abs))

# Ex 4. Step (Heaviside).
f_step = Heaviside(t)
F_step = fourier_transform(f_step, t, omega)
print("\nEx4. Step (Heaviside)")
print(pretty(F_step))

# Ex 5. Cosine.
f_cos = a * cos(b * t)
F_cos = fourier_transform(f_cos, t, omega)
print("\nEx5. Cosine")
print(pretty(F_cos))

# Ex 6. Sign.
f_sign = sign(t)
F_sign = fourier_transform(f_sign, t, omega)
print("\nEx6. Sign")
print(pretty(F_sign))

# Ex 7. Right-side exponential.
f_exp = exp(-t * Abs(a)) * Heaviside(t)
F_exp = a / (1j * omega + b)
print("\nEx7. Right-side exponential")
print(pretty(F_exp))
