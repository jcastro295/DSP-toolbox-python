""" 
spec.py
-------
"""

import numpy as np

def spectrogram_coded(x, window, noverlap, nfft, fs):
    """
    This function uses the same Matlab implementation to return the power spectral density from the time series x
    Args:
        x (np.ndarray): Time series
        fs (float): sampling frequency, Hz
        nfft (int): number of samples for each FFT
            frequency bin size, df = nfft/fs
        window (np.ndarray): Time window. Use window to divide the signal into segments.

    Returns:
        Gxx : ndarray of float
            PSD in in linear units
        t : ndarray of float
            t is the time array corresponding to the time steps
        f : ndarray of float
            f is the frequency array
    """

    x = np.squeeze(x.astype(np.float32))     # Format the input time series
    W = np.mean(np.multiply(window, window)) # normalizing factor
    nx = len(window)  # Window size
    time_step = nx/fs # Time spacing
    total_time = len(x)/fs
    # determine how many unique times will be in the final spectrogram
    perc_noverlap = noverlap/nx             # Overlap in percentage 0-99
    time_step = time_step*(1-perc_noverlap) # Time spacing
    nt = np.floor(total_time/time_step)   # Number of time points
    # make time array of final spectrogram
    t = np.arange(1,nt+1)*time_step + time_step
    # make nt an integer
    nt = int(nt.item())
    f = np.fft.fftfreq(nfft, d=1/fs)[:int(np.ceil(nfft/2)+1)]
    # number of frequencies returned from the FFT
    nf = len(f)

    Gxx_all = np.zeros([nt,nf])
    nstart=0

    for time_idx in np.arange(nt):
        # beginning at index nstart sample, do the FFT on a signal ns samples in length
        nstop = nstart + nx

        x1 = x[int(nstart):int(nstop)]

        if len(x1) == nx:
            # apply the window
            x1 = np.multiply(x1.T, window)
            if nx > nfft: # If the number of time steps is greater than nfft then do a cyclic average
                n_aux = int(np.ceil(nx/nfft)) 
                new_size = n_aux*nfft
                x1 = np.resize(x1, new_size)
                if new_size > nx:
                    x1[nx:] = 0
                x1 = x1.reshape(n_aux,nfft)
                x1 = np.sum(x1,0)
                pass

            # calculate the fft for a single change and the time_idx time sample
            # and extract the single sided part
            Xss = np.fft.fft(x1, nfft)[:int(nfft/2)+1]

            # create scale value
            Scale = (2/nfft/fs/W)/(nx/nfft)

            # multiply by our scale value and force the dtype back to original dtype
            Gxx1 = Scale * np.multiply(np.conj(Xss), Xss)
    
            Gxx_all[time_idx] = np.real(Gxx1)

            # update nstart to the next time step
            nstart = nstart + nx - noverlap
        else:
            Gxx_all = Gxx_all[0:nt-3,:]
            t = t[0:nt-3]
    Gxx_all[:,0] = Gxx_all[:,0]/2     # Unscale the factor of two. Don't double unique Nyquist point
    Gxx_all[:,nf-1] = Gxx_all[:,nf-1]/2 # Unscale the factor of two. Don't double unique Nyquist point

    return Gxx_all, f, t

