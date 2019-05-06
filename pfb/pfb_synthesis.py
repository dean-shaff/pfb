import logging

import numpy as np
import scipy.fftpack
import partialize

from .util import os_factor_type
from .rational import Rational

module_logger = logging.getLogger(__name__)

__all__ = [
    "pfb_synthesize"
]


@partialize.partialize
def pfb_synthesize(input_data: np.ndarray,
                   input_fft_length: int = 1024,
                   *,
                   fir_filter_coeff: np.ndarray,
                   apply_deripple: bool,
                   os_factor: os_factor_type) -> np.ndarray:
    """

    Args:
        input_data (np.ndarray): Should be (ndat, nchan) dimensions
    """
    os_factor = Rational.from_str(os_factor)

    ndat, nchan = input_data.shape

    input_os_keep = os_factor.normalize(input_fft_length)
    input_os_discard = input_fft_length - input_os_keep

    output_fft_length = os_factor.normalize(input_fft_length)*nchan

    input_slice = slice(None)
    output_slice = slice(None)

    nblocks = int(ndat / input_fft_length)

    output_data = np.zeros((output_fft_length*nblocks))

    for idx in range(nblocks):
        input_slice.start = idx*input_fft_length
        input_slice.stop = input_slice.start + input_fft_length
        output_slice.start = idx*output_fft_length
        output_slice.stop = output_slice.start + output_fft_length

        chunk = input_data[input_slice, :]
        chunk_fft = np.fft.fftshift(
            scipy.fftpack.fft(chunk, axis=0), axis=0)
        chunk_fft_keep = chunk_fft[input_os_discard:-input_os_discard, :]
        assembled_spectrum = chunk_fft_keep.reshape((output_fft_length, 1))
        if apply_deripple:
            pass
        output_data[output_slice] = assembled_spectrum

    return output_data
