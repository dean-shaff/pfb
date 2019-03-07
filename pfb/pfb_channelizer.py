import os
import time
import logging
import argparse

import numpy as np
import numba

from .util import (
    load_matlab_filter_coef,
    get_most_recent_data_file,
    add_filter_info_to_header
)
from .formats import DADAFile, DataFile
from .rational import Rational

module_logger = logging.getLogger(__name__)


@numba.njit(cache=True, fastmath=True)
def apply_filter(signal: np.ndarray,
                 filter_coef: np.ndarray,
                 filtered: np.ndarray,
                 downsample_by: int,
                 increment_by: int = None) -> np.ndarray:
    """
    filter signal with filter_coef.

    Implementation is a little strange from a numpy perspective, as numba
    doesn't support 2d boolean array indexing.
    """
    if increment_by is None:
        increment_by = downsample_by
    init_filter_size = filter_coef.shape[0]
    # print(f"init_filter_size: {init_filter_size}")
    filter_dtype = filter_coef.dtype
    signal_dtype = signal.dtype
    is_real = True
    if signal_dtype is np.complex64:
        is_real = False

    rem = init_filter_size % downsample_by
    # print(f"rem: {rem}")
    if rem != 0:
        filter_coef_padded = np.zeros(
            (init_filter_size + downsample_by - rem),
            dtype=filter_dtype
        )
        # print(filter_coef_padded.shape)
        # filter_coef_padded[init_filter_size:] = filter_coef
        filter_coef_padded[:init_filter_size] = filter_coef
        filter_coef = filter_coef_padded

    window_size = filter_coef.shape[0]

    down_sample_filter_elem = int(window_size / downsample_by)
    filter_idx = np.arange(window_size).reshape(
        (down_sample_filter_elem, downsample_by))
    # filter_coeff_2d = filter_coef[filter_idx]
    filter_coeff_2d = np.zeros(filter_idx.shape, dtype=filter_dtype)
    for i in range(downsample_by):
        filter_coeff_2d[:, i] = filter_coef[filter_idx[:, i]]

    signal_padded = np.zeros(
        (window_size + signal.shape[0]),
        dtype=signal_dtype
    )
    signal_padded[window_size:] = signal

    down_sample_signal_elem = filtered.shape[0]
    signal_chunk_2d = np.zeros(filter_idx.shape, dtype=signal_dtype)

    for i in range(down_sample_signal_elem):
        idx = i*increment_by
        signal_chunk = signal_padded[idx:idx + window_size][::-1]
        # filtered[i,:] = np.sum(
        #   signal_chunk[filter_idx] * filter_coeff_2d, axis=0)
        for j in range(downsample_by):
            signal_chunk_2d[:, j] = signal_chunk[filter_idx[:, j]]

        if is_real:
            filtered[i, :] = np.sum(signal_chunk_2d * filter_coeff_2d, axis=0)
        else:
            filtered[i, :] = (
                np.sum(signal_chunk_2d.real * filter_coeff_2d, axis=0) +
                1j*np.sum(signal_chunk_2d.imag * filter_coeff_2d, axis=0)
            )

    return filtered


class PFBChannelizer:
    """
    Attributes:
        input_data (np.ndarray): dimensions are (ndat, nchan, npol)
    """
    def __init__(self,
                 input_data: np.ndarray,
                 fir_filter_coeff: np.ndarray,
                 input_tsamp: float = 1.0):

        self.logger = module_logger.getChild("PFBChannelizer")
        self.input_data = input_data
        self.fir_filter_coeff = fir_filter_coeff
        self._fir_filter_coeff_padded = None
        self.oversampling_factor = None
        self.input_tsamp = input_tsamp

        self.output_data = None
        self.output_data_file = None

        if np.iscomplexobj(self.input_data):
            self._complex_dtype = self.input_data.dtype
            self._float_dtype = (np.float32 if
                                 self._complex_dtype == np.float32
                                 else np.float64)

        else:
            self._float_dtype = self.input_data.dtype
            self._complex_dtype = (np.complex64 if
                                   self._float_dtype == np.float32
                                   else np.complex128)

        self.fir_filter_coeff = self.fir_filter_coeff.astype(
            self._float_dtype)

        self._input_npol = self.input_data.shape[2]
        self._input_nchan = self.input_data.shape[1]
        self._input_ndim = 2 if np.iscomplexobj(self.input_data) else 1

        self._input_ndat = self.input_data.shape[0]
        self._input_samples = 0

        self._output_npol = 2
        self._output_nchan = None
        self._output_ndim = 2
        self._output_samples = 0

        self._n_series = None
        self._ndim_ratio = Rational(self._output_ndim, self._input_ndim)

        self._pfb_input_mask = None
        self._pfb_output_mask = None

        self.logger.debug(
            ("PFBChannelizer.__init__: input (ndat, nchan, npol, ndim) = "
             f"({self._input_ndat}, {self._input_nchan}, "
             f"{self._input_npol}, {self._input_ndim})"))

    def pad_fir_filter_coeff(self, nchan: int):
        fir_len = self.fir_filter_coeff.shape[0]
        rem = fir_len % nchan
        self._fir_filter_coeff_padded = np.zeros(
            fir_len + rem,
            dtype=self._float_dtype)
        self._fir_filter_coeff_padded[:fir_len] = \
            self.fir_filter_coeff

        input_mask_dtype = (self._float_dtype if
                            self._input_ndim == 2 else
                            self._complex_dtype)

        self._pfb_input_mask = np.zeros(
            (self._output_npol, self._fir_filter_coeff_padded.shape[0]),
            dtype=input_mask_dtype)

        self._pfb_output_mask = np.zeros(
            (self._output_npol, nchan),
            dtype=self._float_dtype)

        # self.logger.debug(
        #     (f"pad_fir_filter_coeff: self.fir_filter_coeff.dtype: "
        #      f" {self.fir_filter_coeff.dtype}"))
        # self.logger.debug(
        #     (f"pad_fir_filter_coeff: self.fir_filter_coeff.shape:"
        #      f" {self.fir_filter_coeff.shape}"))

    def calc_output_tsamp(self) -> float:
        return (int(self._ndim_ratio) *
                self.input_tsamp *
                self._output_nchan *
                self.oversampling_factor.de / self.oversampling_factor.nu)

    def _init_output_data(self):

        ndat_input = self.input_data.shape[0]
        norm_chan = self.oversampling_factor.normalize(self._output_nchan)
        # int_ndim_ratio = int(self._ndim_ratio)
        self._output_samples = int(ndat_input / norm_chan)
        self._input_samples = self._output_samples * norm_chan

        self.logger.debug(
            f"_init_output_data: ndat_input: {ndat_input}")
        self.logger.debug(
            f"_init_output_data: self._output_samples: {self._output_samples}")
        self.logger.debug(
            f"_init_output_data: self._input_samples: {self._input_samples}")

        # self.output_data = np.zeros((
        #     int(self._output_samples / (self._input_npol * self._input_ndim)),
        #     self._output_nchan,
        #     self._output_ndim * self._output_npol
        # ), dtype=self._float_dtype)

        self.output_data = np.zeros((
            self._output_samples,
            self._output_nchan,
            self._output_npol
        ), dtype=self._complex_dtype)

        self.logger.debug(
            (f"_init_output_data: "
             f"self.output_data.shape: {self.output_data.shape}"))

    def _init_output_data_file(self):

        self.output_data_file = DADAFile(self.output_file_path)

        os_factor = self.oversampling_factor

        self.output_data_file['NDIM'] = 2
        self.output_data_file['NCHAN'] = self._output_nchan
        self.output_data_file['NPOL'] = self._output_npol
        self.output_data_file['TSAMP'] = self.calc_output_tsamp()
        self.output_data_file['OS_FACTOR'] = f"{os_factor.nu}/{os_factor.de}"

    def _dump_data(self, data_file: DataFile):
        # add filter info to the data_file
        fir_info = [{
            "OVERSAMP": str(self.oversampling_factor),
            "COEFF": self.fir_filter_coeff,
            "NCHAN_PFB": self._output_nchan
        }]
        data_file.header = add_filter_info_to_header(
            data_file.header, fir_info)

        data_file.dump_data()

    def _spiral_roll(self, arr: np.ndarray, n: int = None):
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

    def _channelize(self,
                    input_samples: np.ndarray):

        t_total = time.time()

        nchan = self._output_nchan

        output_filtered = np.zeros(
            (self.output_data.shape[0], nchan),
            dtype=input_samples.dtype
        )

        nchan_norm = self.oversampling_factor.normalize(nchan)
        for p in range(self._input_npol):
            t0 = time.time()

            output_filtered = apply_filter(
                input_samples[:, p].copy(),
                self._fir_filter_coeff_padded,
                output_filtered,
                nchan,
                nchan_norm
            )

            self.logger.debug(
                (f"_channelize: "
                 f"Call to filter took {time.time()-t0:.4f} seconds"))

            if self.oversampled:
                t0 = time.time()
                output_filtered = self._spiral_roll(output_filtered, nchan)

                self.logger.debug(
                    (f"_channelize: "
                     f"Shifting array took {time.time()-t0:.4f} seconds"))

            yield output_filtered

            output_filtered_fft = (nchan**2)*np.fft.ifft(
                output_filtered, n=nchan, axis=1)

            self.output_data[:, :, p] = output_filtered_fft

        self.logger.debug(
            (f"_channelize: "
             f"Took {time.time() - t_total:.4f} seconds to channelize"))

    def _prepare_channelize(self,
                            nchan,
                            oversampling_factor,
                            output_file_path=None):
        """
        Do any operations necessary to prepare input and output data structures
        for channelization
        """
        self._output_nchan = nchan

        if hasattr(oversampling_factor, "endswith"):
            oversampling_factor = Rational(*oversampling_factor.split("/"))
        self.oversampling_factor = oversampling_factor
        self.oversampled = False
        if float(self.oversampling_factor) != 1.0:
            self.oversampled = True

        if output_file_path is None:
            os_text = (f"os_{self.oversampling_factor.nu}-"
                       f"{self.oversampling_factor.de}")
            output_file_name = (f"pfb.{os_text}."
                                f"nchan_{nchan}."
                                f"ntaps_{len(self.fir_filter_coeff)}.dump")
            output_file_path = os.path.join(os.getcwd(), output_file_name)

        self.output_file_path = output_file_path

        if self.input_data is None:
            self._load_input_data()
        if self.output_data_file is None:
            self._init_output_data_file()
        if self.output_data is None:
            self._init_output_data()
        self.pad_fir_filter_coeff(nchan)

        input_samples = self.input_data[:self._input_samples, 0, :]
        self.logger.debug((f"_prepare_channelize: "
                           f"input_samples.shape={input_samples.shape}"))

        # do any downsampling necessary for conversion
        # from real to complex data.
        if int(self._ndim_ratio) != 1:
            input_samples = input_samples[::int(self._ndim_ratio), :]

        return input_samples

    def channelize(self, *args, **kwargs):
        input_samples = self._prepare_channelize(*args, **kwargs)
        # if not self.oversampled:
        #     g = self._channelize_fft(*prepped)
        # else:
        g = self._channelize(input_samples)

        for i in g:
            pass
        self.output_data_file.data = self.output_data
        self._dump_data(self.output_data_file)

    @staticmethod
    def from_input_files(input_file_path: str, fir_file_path: str):

        dada_file = DADAFile(input_file_path)
        dada_file.load_data()
        module_logger.debug(
            (f"PFBChannelizer.from_input_files: "
             f"dada_file.ndat={dada_file.ndat}, "
             f"dada_file.npol={dada_file.npol}, "
             f"dada_file.ndim={dada_file.ndim}, "
             f"dada_file.nchan={dada_file.nchan}"))
        input_data = dada_file.data
        module_logger.debug(
            ("PFBChannelizer.from_input_files: "
             f"input_data.dtype={input_data.dtype}"))
        input_tsamp = float(dada_file["TSAMP"])

        fir_config, fir_filter_coef = load_matlab_filter_coef(fir_file_path)

        channelizer = PFBChannelizer(input_data, fir_filter_coef, input_tsamp)

        return channelizer


def create_parser():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    config_dir = os.getenv("PFB_CONFIG_DIR",
                           os.path.join(current_dir, "config"))
    data_dir = os.getenv("PFB_DATA_DIR",
                         os.path.join(current_dir, "data"))

    most_recent_data_file = ""
    try:
        most_recent_data_file = get_most_recent_data_file(data_dir)
    except (IndexError, IOError) as err:
        pass

    parser = argparse.ArgumentParser(
        description="channelize data")

    parser.add_argument("-i", "--input-file",
                        dest="input_file_path",
                        default=most_recent_data_file)

    parser.add_argument("-f", "--fir-file",
                        dest="fir_file_path",
                        default=os.path.join(config_dir,
                                             "OS_Prototype_FIR_8.mat"))

    parser.add_argument("-v", "--verbose",
                        dest="verbose", action="store_true")

    parser.add_argument("-c", "--channels",
                        dest="channels", default=8, type=int)

    parser.add_argument("-os", "--oversampling_factor",
                        dest="oversampling_factor", default="1/1", type=str)

    return parser


if __name__ == "__main__":
    parsed = create_parser().parse_args()
    log_level = logging.INFO
    if parsed.verbose:
        log_level = logging.DEBUG

    logging.basicConfig(level=log_level)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    if parsed.input_file_path == "":
        raise RuntimeError("Need to provide a file to read")

    channelizer = PFBChannelizer.from_input_files(
        parsed.input_file_path,
        parsed.fir_file_path
    )

    channelizer.channelize(parsed.channels, parsed.oversampling_factor)