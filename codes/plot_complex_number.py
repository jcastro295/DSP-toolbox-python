"""
plot_complex_number.py
----------------------

This script demonstrates how to plot a complex number in Cartesian and polar
coordinates.
"""

import numpy as np
import matplotlib.pyplot as plt
import cmath

# Define the complex number
z = 4 + 3j

# Plot in Cartesian coordinates
plt.figure(figsize=(11, 5))
plt.subplot(1, 2, 1)
plt.plot(np.real(z), np.imag(z), 'ro-', linewidth=2, markersize=8)
plt.plot([0, np.real(z)], [0, np.imag(z)],
         color='#1f77b4', linewidth=1.5)
plt.plot([np.real(z), np.real(z)], [0, np.imag(z)], '--',
         color='#7f7f7f')
plt.plot([0, np.real(z)], [np.imag(z), np.imag(z)], '--',
         color='#7f7f7f')
plt.xlabel('Re{z}')
plt.ylabel('Im{z}')
plt.title(f'Complex Number in Cartesian Coordinates')
plt.grid(True)

# Plot in polar coordinates
plt.subplot(1, 2, 2, polar=True)
plt.polar([0, cmath.phase(z)], [0, abs(z)], marker='o', linewidth=2, markersize=8)
plt.title(f'Complex Number in Polar Coordinates\n|z|={abs(z):2.0f}, ∠(z)={np.degrees(cmath.phase(z)):2.2f}°')

# Adjust layout to prevent overlap of titles and labels
plt.tight_layout()

# Display the plots
plt.show()
