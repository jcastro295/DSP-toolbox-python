"""
plot_continuous_discrete_signal.py
----------------------------------

This script demonstrates how to plot continuous, continuous with sampling, and
discrete signals.
"""

import numpy as np
import matplotlib.pyplot as plt

# Continuous signal
t = np.arange(0, 2 * np.pi, np.pi/100)
y_continuous = np.sin(t)

# Continuous signal with sampling
t_sampled = np.arange(0, 2 * np.pi, np.pi/100)
y_sampled = np.sin(t_sampled)
sample_indices = np.arange(0, len(y_sampled), 10)

# Discrete signal
t_discrete = np.arange(0, 2 * np.pi, np.pi / 10)
y_discrete = np.sin(t_discrete)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(t, y_continuous)
plt.title('Continuous signal', fontsize=14)
plt.xlabel('t', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)

plt.subplot(1, 3, 2)
plt.plot(t_sampled, y_sampled, '-o', markevery=sample_indices, markeredgecolor='r')
plt.title('Continuous signal sampling', fontsize=14)
plt.xlabel('t', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)

plt.subplot(1, 3, 3)
plt.stem(t_discrete, y_discrete)
plt.title('Discrete signal', fontsize=14)
plt.xlabel('(nT)', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)

plt.tight_layout()
plt.show()
