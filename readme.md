# Digital Signal Processing Toolbox (python)

This repository contains python codes I developed for teaching an undergraduate course on digital signal processing.

The repository includes codes for:

- Plotting complex signals (magnitude and phase).
- Even-odd signal synthesis.
- Examples of linear and circular convolution.
- Difference equations.
- Forward and inverse Fourier transformations.
- Forward and inverse Laplace transformations.
- Simple FIR and IIR filter implementations.
- Linear and $\mu$-quantization.
- Implementation of DFT and FFT.
- Power spectral density estimation using built-in functions and manual methods.
- Spectrogram plotting and implementation.

## Data Downloading

Some of the codes use additional data for testing. To download the data, use the following command:

```bash
bash download_data.sh
```

## Additional dependencies

Python 3.10 or later is suggested for these codes to avoid any conflicts with functions and packages used.

To install the required Python packages, use pip within your conda or pyenv environment. Install requirements using:

```bash
pip install -r requirements.txt
```

For the pydub package (used for audio reading), ensure you have previously installed ffmpeg or libav on your computer. For further details, visit their [documentation](https://github.com/jiaaro/pydub#dependencies).

I have also written a Matlab version for every code listed here. Please free to check those out [here!](https://github.com/jcastro295/DSP-toolbox-matlab).