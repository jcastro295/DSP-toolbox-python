"""
plot_complex_signal_ex2.py
--------------------------

This script demonstrates how to plot the real, imaginary, magnitude, and phase
parts of a complex signal X(w) = exp(jw)/(exp(jw) - 0.5), 0 <= w <= pi.
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate frequency values from 0 to pi
w = np.linspace(0, np.pi, 501)

# Compute X(w) using the given formula
X = np.exp(1j * w) / (np.exp(1j * w) - 0.5 * np.ones(501))

# Extract magnitude and phase components of X(w)
magX = np.abs(X)
angX = np.angle(X)
realX = np.real(X)
imagX = np.imag(X)

# Plot magnitude part
plt.subplot(2, 2, 1)
plt.plot(w / np.pi, magX)
plt.xlabel('Frequency in pi units')
plt.title('Magnitude Part')
plt.ylabel('Magnitude')
plt.grid()

# Plot angle part
plt.subplot(2, 2, 3)
plt.plot(w / np.pi, angX)
plt.xlabel('Frequency in pi units')
plt.title('Angle Part')
plt.ylabel('Radians')
plt.grid()

# Plot real part
plt.subplot(2, 2, 2)
plt.plot(w / np.pi, realX)
plt.xlabel('Frequency in pi units')
plt.title('Real Part')
plt.ylabel('Real')
plt.grid()

# Plot imaginary part
plt.subplot(2, 2, 4)
plt.plot(w / np.pi, imagX)
plt.xlabel('Frequency in pi units')
plt.title('Imaginary Part')
plt.ylabel('Imaginary')
plt.grid()

# Adjust layout to prevent overlap of titles and labels
plt.tight_layout()

# Display the plots
plt.show()
