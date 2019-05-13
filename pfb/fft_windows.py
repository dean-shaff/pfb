# some common FFT windows for use in FFT based PFB inversion
import logging

import numpy as np
import scipy.signal

__all__ = [
    "no_window",
    "top_hat_window",
    "tukey_window"
]

module_logger = logging.getLogger(__name__)


def no_window(fft_length):
    module_logger.debug(f"no_window: fft_length={fft_length}")
    return np.ones((fft_length, 1))


def top_hat_window(fft_length, overlap):
    module_logger.debug((f"top_hat_window: fft_length={fft_length}, "
                         f"overlap={overlap}"))
    window = np.ones((fft_length, 1))
    if overlap > 0:
        window[:overlap] = 0.0
        window[-overlap:] = 0.0
    return window


def tukey_window(fft_length, overlap):
    module_logger.debug((f"tukey_window: fft_length={fft_length}, "
                         f"overlap={overlap}"))
    window = np.ones((fft_length, 1))
    if overlap > 0:
        hann_portion = scipy.signal.hann(2*overlap)
        window[:overlap, 0] = hann_portion[:overlap]
        window[-overlap:, 0] = hann_portion[-overlap:]

    return window
