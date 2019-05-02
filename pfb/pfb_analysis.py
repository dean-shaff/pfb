import logging
import typing

import numpy as np
import scipy.fftpack
import numba
import partialize

from . import util, rational

module_logger = logging.getLogger(__name__)


__all__ = [
    "pfb_analysis",
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


@numba.njit(cache=True, fastmath=True)
def _apply_filter(signal: np.ndarray,
                  filter_coeff: np.ndarray,
                  filtered: np.ndarray,
                  downsample_by: int,
                  signal_is_real: bool = False,
                  increment_by: int = None,
                  input_padding: int = None) -> np.ndarray:
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

    filter_dtype = filter_coeff.dtype
    signal_dtype = signal.dtype

    window_size = filter_coeff.shape[0]

    down_sample_filter_elem = int(window_size / downsample_by)
    filter_idx = np.arange(window_size).reshape(
        (down_sample_filter_elem, downsample_by))
    # filter_coeff_2d = filter_coeff[filter_idx]
    filter_coeff_2d = np.zeros(filter_idx.shape, dtype=filter_dtype)
    for i in range(downsample_by):
        filter_coeff_2d[:, i] = filter_coeff[filter_idx[:, i]]

    if input_padding is None:
        input_padding = window_size

    signal_padded = np.zeros(
        (input_padding + signal.shape[0]),
        dtype=signal_dtype
    )
    signal_padded[input_padding:] = signal

    down_sample_signal_elem = filtered.shape[0]
    signal_chunk_2d = np.zeros(filter_idx.shape, dtype=signal_dtype)

    for i in range(down_sample_signal_elem):
        idx = i*increment_by
        signal_chunk = signal_padded[idx:idx + window_size][::-1]
        # filtered[i,:] = np.sum(
        #   signal_chunk[filter_idx] * filter_coeff_2d, axis=0)
        for j in range(downsample_by):
            signal_chunk_2d[:, j] = signal_chunk[filter_idx[:, j]]

        if signal_is_real:
            filtered[i, :] = np.sum(signal_chunk_2d * filter_coeff_2d, axis=0)
        else:
            filtered[i, :] = (
                np.sum(signal_chunk_2d.real * filter_coeff_2d, axis=0) +
                1j*np.sum(signal_chunk_2d.imag * filter_coeff_2d, axis=0)
            )

    return filtered


@partialize.partialize
def calc_output_tsamp(input_tsamp: float,
                      input_ndim: int = 2,
                      output_ndim: int = 2,
                      *,
                      nchan: int,
                      os_factor: typing.Any) -> float:
    os_factor = rational.Rational.from_str(os_factor)
    ndim_ratio = int(output_ndim / input_ndim)
    output_tsamp = float(input_tsamp * ndim_ratio * nchan *
                         os_factor.de / os_factor.nu)
    return output_tsamp


@partialize.partialize
def pfb_analysis(input_data: np.ndarray,
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

    module_logger.debug(f"pfb_analysis: input_data.shape={input_data.shape}")
    module_logger.debug(f"pfb_analysis: fir_filter_coeff.shape={fir_filter_coeff.shape}")
    module_logger.debug(f"pfb_analysis: nchan={nchan}")
    module_logger.debug(f"pfb_analysis: os_factor={os_factor}")

    # t_total = time.time()

    ndat_input = input_data.shape[0]
    nchan_norm = os_factor.normalize(nchan)

    output_samples = int(ndat_input / nchan_norm)
    input_samples = output_samples * nchan_norm

    module_logger.debug(f"pfb_analysis: input_samples={input_samples}")
    module_logger.debug(f"pfb_analysis: output_samples={output_samples}")


    fir_filter_coeff_padded = _pad_filter(fir_filter_coeff, nchan)

    output_filtered = np.zeros((
        output_samples - int((fir_filter_coeff.shape[0] - 1)/2),
        nchan
    ), dtype=output_dtype)

    # t0 = time.time()
    output_filtered = _apply_filter(
        input_data[:input_samples],
        fir_filter_coeff_padded,
        output_filtered,
        nchan,
        signal_is_real=not np.iscomplexobj(input_data),
        increment_by=nchan_norm,
        input_padding=0
    )

    # module_logger.debug(
    #     (f"pfb_analysis: "
    #      f"Call to filter took {time.time()-t0:.4f} seconds"))

    if float(os_factor) > 1.0:
        # t0 = time.time()
        output_filtered = _spiral_roll(output_filtered, nchan)
        # module_logger.debug(
        #     (f"pfb_analysis: "
        #      f"Shifting array took {time.time()-t0:.4f} seconds"))

    # yield output_filtered

    output_filtered_fft = nchan*scipy.fftpack.fft(
        output_filtered, n=nchan, axis=1)

    # output_filtered_fft = (nchan**2)*scipy.fftpack.ifft(
    #     output_filtered, n=nchan, axis=1)
    # module_logger.debug(
    #     (f"pfb_analysis: "
    #      f"Took {time.time() - t_total:.4f} seconds to channelize"))

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
