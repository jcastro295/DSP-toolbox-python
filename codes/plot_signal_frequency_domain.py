"""
plot_signal_frequency_domain.py
-------------------------------

This script demonstrates how to plot the magnitude and phase parts of a complex
signal x(t) = 0.00772 + 0.122j - 0.08866 + 0.2805j + 0.48 - 0.08996j + 
0.01656 - 0.1352j + 0.04724, 0 <= t <= 2*pi.
"""

import numpy as np
import matplotlib.pyplot as plt


f_k = np.arange(100, 1800, 100)
a_k = [0, 0.00772 + 0.122j, 0, -0.08866 + 0.2805j, 0.48 - 0.08996j,
       *[0] * 10, 0.01656 - 0.1352j, 0.04724]

# The magnitude (a) is an even function
# with respect to f = 0; the phase (b) is odd since
# a_{−k} = a^{∗}_k

a = np.concatenate((np.flip(np.conj(a_k)), [0], a_k))
f = np.concatenate((-np.flip(f_k), [0], f_k))

plt.figure(figsize=(8, 6))

plt.subplot(2, 1, 1)
plt.stem(f, np.abs(a), linefmt='b-', markerfmt='bo')
plt.title('(a) Magnitude spectrum', fontsize=14)
plt.ylabel('|a_k|', fontsize=12)

plt.subplot(2, 1, 2)
plt.stem(f, np.angle(a), linefmt='b-', markerfmt='bo')
plt.title('(b) Phase angle spectrum (rad)', fontsize=14)
plt.xlabel('Frequency f (Hz)', fontsize=12)
plt.ylabel('angle(a_k) (rad)', fontsize=12)

plt.tight_layout()

# generating table
print('k\tf_k (Hz)\ta_k\t\t\tMag\t\tPhase')
for k in range(1, len(a_k) + 1):
    print(f'{k}\t{f_k[k-1]}\t{np.real(a_k[k-1]):.4f}+{np.imag(a_k[k-1]):.4f}j\t\t{np.abs(a_k[k-1]):.4f}\t\t{np.angle(a_k[k-1]):.4f}')

plt.show()