"""
plot_signal_harmonics.py
-------------------------

This script demonstrates how to plot the sinusoidal components of a complex signal
x(t) = 0.0771*exp(j2*pi*100*t) - 0.8865*exp(j2*pi*200*t) + 4.8001*exp(-j2*pi*250*t) 
+ 0.1657*exp(-j2*pi*1600*t) + 0.4723*exp(-j2*pi*1700*t)
"""

import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

fs = 10000
ZZ = 10000 * np.array([0.0771 + 1j*1.2202, -0.8865 + 1j*2.8048, 4.8001 - 1j*0.8995, 0.1657 - 1j*1.3520, 0.4723])
Fo = 100
ff = Fo * np.array([2, 4, 5, 16, 17])
dur = 4 / Fo
tt = np.arange(0, dur + 1/fs, 1/fs)
xi = np.exp(-2j * np.pi * tt[:, np.newaxis] * ff)
xx = np.dot(xi, np.array(ZZ))

dur_long = 1.5
ttt = np.arange(0, dur_long + 1/fs, 1/fs)  # Time vector for the longer duration

# Convert time to milliseconds for plotting
tms = tt * 1000

# Plot and play the sinusoidal components
xxx = np.zeros(len(ttt))
fig = plt.figure(figsize=(8, 10))

for i in range(5):
    ax = fig.add_subplot(5, 1, i + 1)
    x_i = np.dot(xi[:, :i+1], ZZ[:i+1])
    ax.plot(tms, np.real(x_i), linewidth=2)
    ax.set_title(f'{i + 1} Harmonic(s)', fontsize=14)
    
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_xlim(0, np.max(tms))

    # Play the sound
    xx = np.real(np.exp(-2j*np.pi*ttt.T*ff[i])*ZZ[i])
    xxx = xxx + xx
    sd.play(np.real(xxx)/max(abs(np.real(xxx))), fs)
    sd.wait()
    fig.canvas.draw_idle()
    input(f'{i+1} harmonics played. Press Enter to continue...')

plt.xlabel('Time (ms)', fontsize=12)
plt.tight_layout()
plt.show()