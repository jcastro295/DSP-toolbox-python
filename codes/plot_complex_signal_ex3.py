"""
plot_complex_signal_ex3.py
--------------------------

This script demonstrates how to plot the magnitude and phase parts of a complex
signal z(t) = exp(j(omega1*t+phi)) + exp(j(omega2*t+phi)), 0 <= t <= 2*pi.
"""

import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 2*np.pi, 100)
f1 = 0.1
f2 = 0.5
omega1 = 2*np.pi*f1
omega2 = 2*np.pi*f2
phi = 0
z = np.exp(1j*(omega1*t+phi)) + np.exp(1j*(omega2*t+phi))

# Plot magnitude of complex signal
plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
plt.plot(t, np.abs(z), color='#d62728', linewidth=1.5)
plt.grid(True)
plt.title('Magnitude')
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.xticks(np.arange(0, 2*np.pi+0.5, np.pi/2), ['0', '$\\frac{\\pi}{2}$', '$\\pi$', '$\\frac{3\\pi}{2}$', '$2\\pi$'])
plt.yticks(np.arange(0, 2.5, 0.5))
plt.ylim(0, 2.2)

# Plot phase of complex signal
plt.subplot(2, 1, 2)
plt.plot(t, np.angle(z), color='#d62728', linewidth=1.5)
plt.grid(True)
plt.title('Phase')
plt.xlabel('Time')
plt.ylabel('Phase')
plt.xticks(np.arange(0, 2*np.pi+0.5, np.pi/2), ['0', '$\\frac{\\pi}{2}$', '$\\pi$', '$\\frac{3\\pi}{2}$', '$2\\pi$'])
plt.yticks(np.arange(-np.pi, np.pi+0.5, np.pi/2), ['$-\\pi$', '$-\\frac{\\pi}{2}$', '0', '$\\frac{\\pi}{2}$', '$\\pi$'])
plt.ylim(-np.pi, np.pi)
plt.tight_layout()

plt.show()
