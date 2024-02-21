"""

quant_bits.py
-------------

"""

import numpy as np

def quant_bits(input_signal, N):
    """
    Quantizes the input signal according to the bit depth N.
    
    Args:
    input_signal (numpy.ndarray): Input signal to be quantized.
    N (int): Bit depth indicating the number of bits used for quantization.

    Returns:
    Q (numpy.ndarray): Quantized signal.
    q (float): Quantization step size.
    """
    
    # Find the maximum and minimum values in the input signal
    max_peak = np.max(input_signal)
    min_peak = np.min(input_signal)

    # Calculate the quantization step size
    q = (max_peak - min_peak) / (2 ** N)

    # Quantize the input signal
    Q = np.round(input_signal / q) * q

    return Q, q
