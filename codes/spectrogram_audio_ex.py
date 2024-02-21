"""
spectrogram_audio_ex.py
------------------------

This script computes the spectrogram of an audio signal using the FFT-based
method. The audio signal is read from a .wav file and is normalized to the range
[-1, 1]. The spectrogram is computed using the FFT-based method and is plotted.
"""

import os
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from matplotlib.animation import FuncAnimation
from pydub import AudioSegment
from scipy.signal import resample_poly, spectrogram


data_folder = '../data'

# Read the audio file
sound = AudioSegment.from_wav(os.path.join(data_folder, 'chirping_sound.wav'))
fs = sound.frame_rate
left = sound.split_to_mono()[0]
x = np.array(left.get_array_of_samples())
x = x / sound.max_possible_amplitude  # Normalize the data

# using a single channel 
x = x[:int(0.1*len(x))]

x_not_sampled = x.copy()
fs_not_sampled = fs
# resample signal 
resample_fs = 25000
x = resample_poly(x, resample_fs, fs)

fs = resample_fs

N = len(x)     #  Number of time samples in x
t = np.arange(N)/fs

#  Calculate spectrograms 
t_window = 0.05                 #  Integration time in seconds, ideally it is ns/fs
n_window = fs*t_window;         #  Window size, i.e., number of points to be considered in the FFT
window = np.hamming(n_window);  #  Apply the window you would like to use.
overlap = 0.9                   #  Overlap (0.0 - 0.99).Typical value: 0.5
ns = n_window                   #  number of samples for each FFT

#  The final time step will be: t_window*(1-overlap) in [sec]
#  The frequency step is: fs/ns
n_overlap = np.round(n_window*overlap)   # Number of overlap points

[F, T, P] = spectrogram(x, fs=fs, window=window, noverlap=n_overlap,
                       nfft=ns, detrend=False, mode='psd')
spec = 10*np.log10(np.abs(P)).T #  In dB

# Setup a separate thread to play the music
music_thread = threading.Thread(target=sd.play, args=(x_not_sampled,), 
                                kwargs={'samplerate': fs_not_sampled, 'blocking':True})

fig = plt.figure(figsize=(10, 5), layout='constrained')

ax1 = fig.add_subplot(2,1,1)
ax1.plot(t, x)
h1 = ax1.axvline(t[0], color='r', linewidth=1.5)
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Amplitude', fontsize=12)
ax1.set_title('Time Series', fontsize=14)
ax1.set_xlim(t[0], t[-1])

# Plot spectrogram snapshots (ns samples spaced dt_save apart)
ax2 = fig.add_subplot(2,1,2)
im1 = ax2.pcolormesh(T, F, spec.T, shading='auto', vmin=-150, vmax=-50, cmap='jet')
h2 = ax2.axvline(t[0], color='r', linewidth=1.5)
cbar = plt.colorbar(im1, ax=ax2, label='PSD (dB)', pad=0.01)
ax2.set_xlabel('Time (sec)', fontsize=12)
ax2.set_ylabel('Frequency (Hz)', fontsize=12)
ax2.set_title(f'Spectrogram: {len(F):d} freqs x {len(T):d} times', fontsize=14)
ax2.set_xlim(t[0], t[-1])


# Matplotlib function to initialize animation
def init():
    global t
    h1.set_data([t[0], t[0]], [0,1])
    h2.set_data([t[0], t[0]], [0,1])

    return (h1, h2)

def animate(frame):
    global music_start, t
    if frame == 0:
        music_thread.start()
        music_start = time.time()
    frame = round((time.time() - music_start)/t[-1] * len(t))
    try:
        h1.set_data([t[frame], t[frame]], [0,1])
        h2.set_data([t[frame], t[frame]], [0,1])
    except:
        h1.set_data([t[-1], t[-1]], [0,1])
        h2.set_data([t[-1], t[-1]], [0,1])
    return (h1, h2)

interval = 50 # ms
anim = FuncAnimation(fig, animate, frames=np.arange(np.ceil(t[-1]/(interval*1e-3)+1)), 
                     init_func=init, interval=interval, blit=True, repeat=False)
plt.show()