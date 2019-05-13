import logging

import numpy as np
import scipy.fftpack
import scipy.signal
import partialize

from . import util
from .rational import Rational

module_logger = logging.getLogger(__name__)

__all__ = [
    "calc_input_tsamp",
    "pfb_synthesize"
]


@partialize.partialize
def calc_input_tsamp(output_tsamp: float,
                     input_ndim: int = 2,
                     output_ndim: int = 2,
                     *,
                     nchan: int,
                     os_factor: util.os_factor_type) -> float:
    os_factor = Rational.from_str(os_factor)
    ndim_ratio = int(output_ndim / input_ndim)
    input_tsamp = float((output_tsamp * os_factor.nu) /
                        (ndim_ratio*nchan*os_factor.de))
    return input_tsamp


@partialize.partialize
def pfb_synthesize(input_data: np.ndarray,
                   input_fft_length: int = 1024,
                   input_overlap: util.overlap_type = None,
                   fft_window: np.ndarray = None,
                   *,
                   fir_filter_coeff: np.ndarray,
                   apply_deripple: bool,
                   os_factor: util.os_factor_type) -> np.ndarray:
    """

    Args:
        input_data (np.ndarray): Should be (ndat, nchan) dimensions
    """
    ndat, nchan = input_data.shape
    input_dtype = input_data.dtype
    output_dtype = util.complex_dtype_lookup[input_dtype]
    output_float_dtype = util.float_dtype_lookup[input_dtype]

    os_factor = Rational.from_str(os_factor)
    if input_overlap is None:
        input_overlap = 0
    else:
        if hasattr(input_overlap, "__call__"):
            input_overlap = input_overlap(input_fft_length)

    if fft_window is None:
        fft_window = np.ones((input_fft_length, 1))
    fft_window = fft_window.astype(output_float_dtype)

    output_overlap = os_factor.normalize(input_overlap)*nchan
    output_overlap_slice = slice(0, None)
    if output_overlap != 0:
        output_overlap_slice = slice(output_overlap, -output_overlap)

    input_os_keep = os_factor.normalize(input_fft_length)
    input_os_keep_2 = int(input_os_keep/2)
    input_os_discard = int((input_fft_length - input_os_keep)/2)
    input_keep = input_fft_length - 2*input_overlap

    output_fft_length = os_factor.normalize(input_fft_length)*nchan
    output_keep = output_fft_length - 2*output_overlap

    nblocks = int((ndat - 2*input_overlap) / input_keep)

    filter_response_len = nchan*input_os_keep_2
    filter_response = np.ones(output_fft_length,
                              dtype=output_float_dtype)
    if apply_deripple:
        h = np.abs(scipy.signal.freqz(
            fir_filter_coeff.flatten(), 1, filter_response_len)[1])
        h = h[:input_os_keep_2].astype(output_float_dtype)
        intra_channel_response = np.append(h[::-1], h)
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 1)
        # ax.plot(1/intra_channel_response)
        # plt.show()
        inter_channel_response = np.tile(intra_channel_response, nchan)
        inter_channel_response = np.roll(
            inter_channel_response, -input_os_keep_2)
        filter_response /= inter_channel_response

    output_data = np.zeros((output_keep*nblocks), dtype=output_dtype)

    module_logger.debug(f"pfb_synthesize: input_overlap={input_overlap}")
    module_logger.debug(f"pfb_synthesize: output_overlap={output_overlap}")
    module_logger.debug(f"pfb_synthesize: input_fft_length={input_fft_length}")
    module_logger.debug((f"pfb_synthesize: output_fft_length="
                         f"{output_fft_length}"))
    module_logger.debug(f"pfb_synthesize: input_os_keep={input_os_keep}")
    module_logger.debug(f"pfb_synthesize: input_os_discard={input_os_discard}")
    module_logger.debug(f"pfb_synthesize: input_overlap={input_overlap}")
    module_logger.debug(f"pfb_synthesize: input_data.dtype={input_data.dtype}")
    module_logger.debug(f"pfb_synthesize: output_data.dtype={output_data.dtype}")
    module_logger.debug(f"pfb_synthesize: filter_response.dtype={filter_response.dtype}")
    module_logger.debug(f"pfb_synthesize: fft_window.dtype={fft_window.dtype}")

    for idx in range(nblocks):
        input_slice_start = idx*input_keep
        input_slice_stop = input_slice_start + input_fft_length
        output_slice_start = idx*output_keep
        output_slice_stop = output_slice_start + output_keep

        chunk = input_data[input_slice_start:input_slice_stop, :]
        chunk_fft = np.fft.fftshift(scipy.fftpack.fft(chunk, axis=0))
        chunk_fft *= fft_window
        chunk_fft_keep = chunk_fft[input_os_discard:-input_os_discard, :]
        assembled_spectrum = chunk_fft_keep.T.reshape(output_fft_length)
        #  Rolling ensures that first half channel gets
        #  placed in the correct part of the assembled spectrum.
        assembled_spectrum = np.roll(assembled_spectrum, -input_os_keep_2)
        assembled_spectrum *= filter_response
        assembled_time_series = scipy.fftpack.ifft(
            np.fft.fftshift(assembled_spectrum))/float(os_factor)
        output_data[output_slice_start:output_slice_stop] = \
            assembled_time_series[output_overlap_slice]

    return output_data
