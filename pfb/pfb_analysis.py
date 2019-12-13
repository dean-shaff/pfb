import logging
import typing
import time

import numpy as np
import scipy.fftpack
import numba
import partialize

from . import util, rational

module_logger = logging.getLogger(__name__)


__all__ = [
    "pfb_analyze",
    "calc_output_tsamp"
]


def _spiral_roll(arr: np.ndarray, n: int = None):
    """
    Cyclically shift arr by n
    """
    if n is None:
        n = arr.shape[1]
    for i in range(n):
        arr[i::n, :] = np.roll(
            arr[i::n, :],
            i % n,
            axis=1
        )
    return arr


def _pad_filter(filter_coeff: np.ndarray,
                downsample_by: int) -> np.ndarray:

    init_filter_size = filter_coeff.shape[0]
    rem = init_filter_size % downsample_by
    filter_dtype = filter_coeff.dtype
    if rem != 0:
        filter_coeff_padded = np.zeros(
            (init_filter_size + downsample_by - rem),
            dtype=filter_dtype
        )
        # print(filter_coeff_padded.shape)
        # filter_coeff_padded[init_filter_size:] = filter_coeff
        filter_coeff_padded[:init_filter_size] = filter_coeff
        filter_coeff = filter_coeff_padded

    return filter_coeff


@numba.njit(cache=True)
def _apply_filter(signal: np.ndarray,
                  filter_coeff: np.ndarray,
                  filtered: np.ndarray,
                  downsample_by: int,
                  signal_is_real: bool = False,
                  increment_by: int = None) -> np.ndarray:
    """
    filter signal with filter_coeff.

    Assumes filter_coeff is already padded appropriately

    Implementation is a little strange from a numpy perspective, as numba
    doesn't support 2d boolean array indexing.

    Args:
        signal
        filter_coeff
        filtered
        downsample_by
        increment_by
        input_padding

    Returns:
        np.ndarray

    """
    if increment_by is None:
        increment_by = downsample_by

    window_size = filter_coeff.shape[0]

    down_sample_filter_elem = int(window_size / downsample_by)

    down_sample_signal_elem = filtered.shape[0]

    # roll_idx = np.arange(down_sample_signal_elem)*increment_by
    # roll_idx -= (roll_idx//downsample_by)*downsample_by

    for i in range(down_sample_signal_elem):
        idx = i*increment_by
        signal_chunk = signal[idx:idx + window_size]
        filtered_i = signal_chunk * filter_coeff
        # if signal_is_real:
        #     filtered_i = signal_chunk * filter_coeff
        # else:
        #     filtered_i = (signal_chunk.real * filter_coeff +
        #                   1j*signal_chunk.imag * filter_coeff)
        index = int(idx - (idx//downsample_by)*downsample_by)
        filtered_i = np.roll(filtered_i, index)
        filtered[i, :] = np.sum(filtered_i.reshape((
            down_sample_filter_elem, downsample_by
        )), axis=0)

    return filtered


@partialize.partialize
def calc_output_tsamp(input_tsamp: float,
                      input_ndim: int = 2,
                      output_ndim: int = 2,
                      *,
                      nchan: int,
                      os_factor: util.os_factor_type) -> float:
    os_factor = rational.Rational.from_str(os_factor)
    ndim_ratio = int(output_ndim / input_ndim)
    output_tsamp = float(input_tsamp * ndim_ratio * nchan *
                         os_factor.de / os_factor.nu)
    return output_tsamp


@partialize.partialize
def pfb_analyze(input_data: np.ndarray,
                use_ifft: bool = False,
                *,
                fir_filter_coeff: np.ndarray,
                nchan: int,
                os_factor: typing.Any):
    """
    Do pfb analysis on single channel input data. Does not handle
    multiple polarizations at once.
    """
    os_factor = rational.Rational.from_str(os_factor)

    input_dtype = input_data.dtype
    output_dtype = util.complex_dtype_lookup[input_dtype]

    module_logger.debug(f"pfb_analyze: input_data.shape={input_data.shape}")
    module_logger.debug(f"pfb_analyze: fir_filter_coeff.shape={fir_filter_coeff.shape}")
    module_logger.debug(f"pfb_analyze: nchan={nchan}")
    module_logger.debug(f"pfb_analyze: os_factor={os_factor}")

    t_total = time.time()
    fir_filter_coeff_padded = _pad_filter(fir_filter_coeff, nchan)
    module_logger.debug((f"pfb_analyze: fir_filter_coeff_padded.shape="
                         f"{fir_filter_coeff_padded.shape}"))

    ndat_input = input_data.shape[0]
    nchan_norm = os_factor.normalize(nchan)

    output_samples = int(
        (ndat_input - fir_filter_coeff_padded.shape[0]) / nchan_norm)
    input_samples = output_samples * nchan_norm

    module_logger.debug(f"pfb_analyze: input_samples={input_samples}")
    module_logger.debug(f"pfb_analyze: output_samples={output_samples}")

    output_filtered = np.zeros((
        output_samples,
        nchan
    ), dtype=output_dtype)

    t0 = time.time()
    output_filtered = _apply_filter(
        input_data,
        fir_filter_coeff_padded,
        output_filtered,
        nchan,
        signal_is_real=not np.iscomplexobj(input_data),
        increment_by=nchan_norm
    )

    module_logger.debug(
        (f"pfb_analyze: "
         f"Call to filter took {time.time()-t0:.4f} seconds"))

    if use_ifft:
        output_filtered_fft = nchan**2*scipy.fftpack.ifft(
            output_filtered, n=nchan, axis=1)
    else:
        output_filtered_fft = nchan*scipy.fftpack.fft(
            output_filtered, n=nchan, axis=1)

    module_logger.debug(
        (f"pfb_analyze: "
         f"Took {time.time() - t_total:.4f} seconds to channelize"))

    return output_filtered_fft


# def create_parser():
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#
#     config_dir = os.getenv("PFB_CONFIG_DIR",
#                            os.path.join(current_dir, "config"))
#     data_dir = os.getenv("PFB_DATA_DIR",
#                          os.path.join(current_dir, "data"))
#
#     parser = argparse.ArgumentParser(
#         description="channelize data")
#
#     parser.add_argument("-i", "--input-file",
#                         dest="input_file_path",
#                         required=True)
#
#     parser.add_argument("-f", "--fir-file",
#                         dest="fir_file_path",
#                         default=os.path.join(config_dir,
#                                              "OS_Prototype_FIR_8.mat"))
#
#     parser.add_argument("-v", "--verbose",
#                         dest="verbose", action="store_true")
#
#     parser.add_argument("-c", "--channels",
#                         dest="channels", default=8, type=int)
#
#     parser.add_argument("-os", "--os_factor",
#                         dest="os_factor", default="1/1", type=str)
#
#     return parser
#
#
# if __name__ == "__main__":
#     parsed = create_parser().parse_args()
#     log_level = logging.INFO
#     if parsed.verbose:
#         log_level = logging.DEBUG
#
#     logging.basicConfig(level=log_level)
#     logging.getLogger("matplotlib").setLevel(logging.ERROR)
#     if parsed.input_file_path == "":
#         raise RuntimeError("Need to provide a file to read")
#
#     input_file = psr_formats.DADAFile(parsed.input_file_path).load_data()
#     channelizer = PFBChannelizer.from_input_files(
#         input_file,
#         parsed.fir_file_path
#     )
#
#     channelizer.channelize(parsed.channels, parsed.os_factor)
#     channelizer.dump_file(header_kwargs={
#         "UTC_START": input_file["UTC_START"]
#     })
