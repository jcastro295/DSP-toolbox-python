"""
complex_number_animation.py
---------------------------

This script demonstrates the animation of a complex number in 3D space. 
The complex number is represented as z = exp(iwt+phi), where w is the angular
frequency and phi is the phase. The real and imaginary parts of the complex
number are plotted against time, and the real and imaginary parts are plotted
against each other. The real part is also plotted against the imaginary part
and the time.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
fs = 10  # Sampling frequency (Hz)
ts = np.arange(0, 20, 1/fs)  # time vector (s)
phi = 5  # phase (rad)
omega = 1  # angular frequency (rad/s)

# Initialize the figure and subplots
fig = plt.figure(figsize=(13, 6.5))

ax1 = fig.add_subplot(2, 4, (1,6), projection='3d')
ax1.view_init(20, 90-38)
ax2 = fig.add_subplot(2, 4, 3)
ax3 = fig.add_subplot(2, 4, 4)
ax4 = fig.add_subplot(2, 4, 7)


# Animation loop
for i in range(len(ts)):
    t = ts[:i+1]
    theta = omega * t
    z = np.exp(1j * (theta + phi))

    # 3D plot
    ax1.cla()
    ax1.plot(t, np.real(z), np.imag(z), color='#1f77b4', linewidth=2)
    ax1.plot((np.min(ts)-0.1)*np.ones(len(t)), np.real(z), np.imag(z),
              color='#bcbd22', linewidth=2)
    ax1.plot(t, -1.1*np.ones(len(t)), np.imag(z),
             color='#2ca02c', linewidth=2)
    ax1.plot(t, np.real(z), -1.1*np.ones(len(t)), 
             color='#d62728', linewidth=2)
    ax1.scatter(t[-1], np.real(z[-1]), np.imag(z[-1]), c='#ff7f0e', s=100)
    ax1.set_xlabel('Time', fontsize=14)
    ax1.set_ylabel('Real Axis', fontsize=14)
    ax1.set_zlabel('Imaginary Axis', fontsize=14)
    ax1.set_title('z = exp(iwt+phi): Blue  Re{z}: Red  Im{z}: Green', fontsize=16)
    ax1.set_xlim(np.min(ts) - 0.1, np.max(ts))
    ax1.set_ylim(-1.1, 1)
    ax1.set_zlim(-1.1, 1)

    # Real vs time
    ax4.cla()
    ax4.plot(np.real(z), t, color='#d62728', linewidth=2)
    ax4.set_xlabel('Real Axis', fontsize=14)
    ax4.set_ylabel('Time', fontsize=14)
    ax4.set_xlim(-1, 1)
    ax4.set_ylim(min(ts), max(ts))
    ax4.grid(True)

    # Real vs imaginary
    ax2.cla()
    ax2.plot(np.real(z), np.imag(z), color='#bcbd22', linewidth=2)
    ax2.plot(np.real(z[-1]), np.imag(z[-1]), 'o', 
             markersize=10, markeredgecolor='#ff7f0e', markerfacecolor='#ff7f0e')
    ax2.plot(np.real(z), np.imag(z), linestyle='--', 
             color='#1f77b4', linewidth=2)
    ax2.set_xlabel('Real Axis', fontsize=14)
    ax2.set_ylabel('Imaginary Axis', fontsize=14)
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.grid(True)

    # Time vs imaginary
    ax3.cla()
    ax3.plot(t, np.imag(z), color='#2ca02c', linewidth=2)
    ax3.set_xlabel('Time', fontsize=14)
    ax3.set_ylabel('Imaginary Axis', fontsize=14)
    ax3.set_xlim(np.min(ts), np.max(ts))
    ax3.set_ylim(-1, 1)
    ax3.grid(True)

    plt.tight_layout()
    plt.pause(0.01)

plt.show()
