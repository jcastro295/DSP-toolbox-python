"""
dtft_ex.py
-----------

This script demonstrates the computation of the Discrete Time Fourier Transform (DTFT) 
of a given sequence.
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the sequences
n = np.arange(-1,4)
x = np.arange(1,6)

# Define the frequency range
k = np.arange(501)
w = (np.pi / 500) * k

# Compute DTFT using matrix-vector product
X = np.dot(x[np.newaxis, :], np.exp(-1j * np.pi / 500 * n[:, np.newaxis] * k))

# Extract magnitude and phase components of X(w)
magX = np.abs(X)
angX = np.angle(X)
realX = np.real(X)
imagX = np.imag(X)

# Plot magnitude part
plt.subplot(2, 2, 1)
plt.plot(w / np.pi, magX.T)
plt.xlabel('Frequency in pi units')
plt.title('Magnitude Part')
plt.ylabel('Magnitude')
plt.grid()

# Plot angle part
plt.subplot(2, 2, 3)
plt.plot(w / np.pi, angX.T)
plt.xlabel('Frequency in pi units')
plt.title('Angle Part')
plt.ylabel('Radians')
plt.grid()

# Plot real part
plt.subplot(2, 2, 2)
plt.plot(w / np.pi, realX.T)
plt.xlabel('Frequency in pi units')
plt.title('Real Part')
plt.ylabel('Real')
plt.grid()

# Plot imaginary part
plt.subplot(2, 2, 4)
plt.plot(w / np.pi, imagX.T)
plt.xlabel('Frequency in pi units')
plt.title('Imaginary Part')
plt.ylabel('Imaginary')
plt.grid()

# Adjust layout to prevent overlap of titles and labels
plt.tight_layout()

# Display the plots
plt.show()
