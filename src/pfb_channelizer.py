import os
import time
import logging
import argparse

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import numba

from .util import (
    load_dada_file,
    dump_dada_file,
    load_matlab_filter_coef,
    get_most_recent_data_file
)
from .rational import Rational

module_logger = logging.getLogger(__name__)


@numba.njit(cache=True, fastmath=True)
def filter(signal, filter_coef, filtered, downsample_by, increment_by=None):
    """
    filter signal with filter_coef.

    Implementation is a little strange from a numpy perspective, as numba
    doesn't support 2d boolean array indexing.
    """
    if increment_by is None:
        increment_by = downsample_by
    init_filter_size = filter_coef.shape[0]
    filter_dtype = filter_coef.dtype
    signal_dtype = signal.dtype
    is_real = True
    if signal_dtype is np.complex64:
        is_real = False

    rem = init_filter_size % downsample_by
    if rem != 0:
        filter_coef_padded = np.zeros(
            (init_filter_size + downsample_by - rem),
            dtype=filter_dtype
        )
        filter_coef_padded[init_filter_size:] = filter_coef
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

    def __init__(self, input_file_path, fir_file_path,
                 output_file_path=None):

        self.logger = module_logger.getChild("PFBChannelizer")
        self.input_file_path = input_file_path
        self.fir_file_path = fir_file_path

        self.oversampling_factor = None
        self.output_file_path = None

        self._float_dtype = np.float32
        self._complex_dtype = np.complex64

        self.input_data = None
        self.input_header = None
        self._input_npol = None
        self._input_nchan = 1
        self._input_ndim = 1

        self.output_data = None
        self.output_header = None
        self._output_npol = 2
        self._output_nchan = None
        self._output_ndim = 2

        self._n_series = None
        self._ndim_ratio = Rational(1, 1)
        self._input_samples = 0
        self._output_samples = 0

        self._fir_config = None
        self._fir_filter_coef = None

    def _load_fir_config(self, diagnostic_plots=False, pad=True):

        t0 = time.time()
        if self.fir_file_path.endswith(".mat"):
            self._fir_config, self._fir_filter_coef = load_matlab_filter_coef(
                self.fir_file_path)
            self._fir_filter_coef = self._fir_filter_coef.astype(self._float_dtype)
        else:
            raise RuntimeError(
                ("No support for filter coefficient "
                 "files other than matlab files"))

        if diagnostic_plots:
            plt.ion()
            fig, ax = plt.subplots()
            ax.grid(True)
            ax.plot(self._fir_filter_coef)
            input(">> ")

        rem = self._fir_filter_coef.shape[0] % self._output_nchan
        if rem != 0 and pad:
            self._fir_filter_coef = np.append(
                self._fir_filter_coef,
                np.zeros(self._output_nchan - rem, dtype=self._float_dtype)
            )

        input_mask_dtype = self._float_dtype
        if self._input_ndim == 2:
            input_mask_dtype = self._complex_dtype

        self._pfb_input_mask = np.zeros(
            (self._output_npol, self._fir_filter_coef.shape[0]),
            dtype=input_mask_dtype
        )

        self._pfb_output_mask = np.zeros(
            (self._output_npol, self._output_nchan),
            dtype=self._float_dtype
        )

        self.logger.debug(
            (f"_load_fir_config: self._fir_filter_coef.dtype: "
             f" {self._fir_filter_coef.dtype}"))
        self.logger.debug(
            (f"_load_fir_config: self._fir_filter_coef.shape:"
             f" {self._fir_filter_coef.shape}"))
        self.logger.debug(
            (f"_load_fir_config: Took {time.time()-t0:.4f} "
             f"seconds to load pfb configuration data"))

    def _load_input_data(self, **kwargs):

        t0 = time.time()
        header, data = load_dada_file(
            self.input_file_path, **kwargs
        )

        self.logger.debug(
            (f"_load_input_data: "
             f"Took {time.time()-t0:.4f} seconds to load input data"))

        self._float_dtype = data.dtype
        self._complex_dtype = (np.complex64 if
                               self._float_dtype == np.float32
                               else np.complex128)

        self.input_header = header
        self.input_data = data

        self._input_npol = int(self.input_header["NPOL"])
        self._input_ndim = int(self.input_header["NDIM"])
        self._input_nchan = int(self.input_header["NCHAN"])
        self._ndim_ratio = Rational(self._output_ndim, self._input_ndim)

    def _init_output_data(self):

        if self.input_header is None:
            raise RuntimeError(
                "Need to load input header before initializing output data")

        ndat_input = self.input_data.shape[0]
        norm_chan = self.oversampling_factor.normalize(self._output_nchan)
        int_ndim_ratio = int(self._ndim_ratio)
        self._output_samples = int(ndat_input / (norm_chan * int_ndim_ratio))
        self._input_samples = self._output_samples * norm_chan * int_ndim_ratio

        self.logger.debug(
            f"_init_output_data: ndat_input: {ndat_input}")
        self.logger.debug(
            f"_init_output_data: self._output_samples: {self._output_samples}")
        self.logger.debug(
            f"_init_output_data: self._input_samples: {self._input_samples}")

        self.output_data = np.zeros((
            int(self._output_samples / (self._input_npol * self._input_ndim)),
            self._output_nchan,
            self._output_ndim * self._output_npol
        ), dtype=self._float_dtype)

    def _init_output_header(self):

        if self.input_header is None:
            raise RuntimeError(
                "Need to load input header before initializing output header")

        os_factor = self.oversampling_factor

        self.output_header = self.input_header.copy()
        self.output_header['NDIM'] = 2
        self.output_header['NCHAN'] = self._output_nchan
        self.output_header['NPOL'] = self._output_npol
        # have to adjust TSAMP
        input_tsamp = float(self.input_header["TSAMP"])
        self.logger.debug(f"_init_output_header: input_tsamp: {input_tsamp}")
        output_tsamp = (
            int(self._ndim_ratio) *
            input_tsamp *
            self._output_nchan *
            os_factor.de / os_factor.nu
        )
        self.logger.debug(f"_init_output_header: output_tsamp: {output_tsamp}")
        self.output_header['TSAMP'] = output_tsamp
        self.output_header['OS_FACTOR'] = f"{os_factor.nu}/{os_factor.de}"

    def _dump_data(self, header, data):

        dump_dada_file(self.output_file_path, header, data)

    def _spiral_roll(self, arr, n=None):
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

    def _channelize_fft(self,
                        input_samples,
                        input_samples_per_pol_dim,
                        output_samples_per_pol_dim):
        if self.oversampled:
            raise RuntimeError("_channelize_fft doesn't work in the oversampled case")

        nchan = self._output_nchan

        output_filtered = np.zeros(
            (output_samples_per_pol_dim, nchan),
            dtype=input_samples.dtype
        )
        nchan_norm = self.oversampling_factor.normalize(nchan)

        # window_size = self._fir_filter_coef.shape[0]
        # samples_per_window = int(window_size/nchan)
        # filter_windows = int((output_samples_per_pol_dim*nchan_norm - window_size)/nchan_norm)
        # idx = np.arange(output_samples_per_pol_dim*nchan_norm)
        # os_idx = np.zeros(filter_windows * samples_per_window, dtype=int)
        # for i in range(filter_windows):
        #     ii = samples_per_window*i
        #     os_idx[ii:ii+samples_per_window] = idx[i*nchan_norm:i*nchan_norm + window_size][::nchan]
        # os_idx = np.sort(np.unique(os_idx))

        for p in range(self._input_npol):
            p_idx = self._output_ndim * p
            t0 = time.time()

            input_samples_padded = np.append(
                np.zeros(
                    int(self._fir_filter_coef.shape[0]),
                    dtype=self._float_dtype
                ),
                np.conj(
                    input_samples[:output_samples_per_pol_dim*nchan, p][::-1]
                )
            )
            t0 = time.time()
            for c in range(nchan):
                filter_decimated = self._fir_filter_coef[c::nchan]

                input_decimated = input_samples_padded[c::nchan]
                # if not self.oversampled:
                    # input_decimated = input_samples_padded[c::nchan]
                # else:
                #     input_decimated = input_samples_padded[os_idx + c]
                filtered = scipy.signal.fftconvolve(
                    input_decimated, filter_decimated, "full")

                # no idea why the following is necessary
                if c % 2 != 0:
                    filtered = -filtered

                delta = filtered.shape[0] - output_samples_per_pol_dim
                delta_2 = int(delta/2)
                output_filtered[:output_samples_per_pol_dim, c] = \
                    filtered[delta_2:output_samples_per_pol_dim + delta_2]

            self.logger.debug(
                (f"_channelize_fft: Calls to scipy.signal.fftconvolve "
                 f"took {time.time()-t0:.4f} seconds"))

            # if self.oversampled:
            #     t0 = time.time()
            #     output_filtered = self._spiral_roll(output_filtered, nchan)
            #     self.logger.debug(
            #         (f"_channelize_fft: "
            #          f"Shifting array took {time.time()-t0:.4f} seconds"))

            yield output_filtered

            output_filtered_fft = (nchan**2)*np.fft.ifft(
                output_filtered, n=nchan, axis=1)

            self.output_data[:, :, p_idx] = np.real(output_filtered_fft)
            self.output_data[:, :, p_idx+1] = np.imag(output_filtered_fft)

    def _channelize(self,
                    input_samples,
                    input_samples_per_pol_dim,
                    output_samples_per_pol_dim):

        t_total = time.time()

        nchan = self._output_nchan

        filter_coef_per_chan = int(self._fir_filter_coef.shape[0] / nchan)

        output_filtered = np.zeros(
            (output_samples_per_pol_dim, nchan),
            dtype=input_samples.dtype
        )

        # output_filtered = output_filtered.copy()

        nchan_norm = self.oversampling_factor.normalize(nchan)

        for p in range(self._input_npol):
            p_idx = self._output_ndim * p
            t0 = time.time()

            output_filtered = filter(
                input_samples[:, p].copy(),
                self._fir_filter_coef,
                output_filtered,
                nchan,
                nchan_norm
            )

            # output_filtered = output_filtered[filter_coef_per_chan:, :]
            # output_filtered[:filter_coef_per_chan, :] = 0.0

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

            # output_filtered_fft = nchan*np.fft.fft(
            #     output_filtered, n=nchan, axis=1)

            self.output_data[:, :, p_idx] = np.real(output_filtered_fft)
            self.output_data[:, :, p_idx+1] = np.imag(output_filtered_fft)

        # print(self.output_data.shape)
        # self.output_data = self.output_data[int(filter_coef_per_chan/2):, :, :]
        # self.output_header['OFFSET'] = filter_coef_per_chan * nchan * self._output_ndim * self._output_npol * np.dtype(self._float_dtype).itemsize
        # print(self.output_data.shape)

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
            os_text = "cs"
            if self.oversampled:
                os_text = "os"
            output_file_name = "py_channelized." + os.path.basename(self.input_file_path)
            output_file_path = os.path.join(
                os.path.dirname(self.input_file_path), output_file_name)
            split = output_file_path.split(".")
            split.insert(-1, os_text)
            output_file_path = ".".join(split)

        self.output_file_path = output_file_path

        if self.input_data is None:
            self._load_input_data()
        if self.output_header is None:
            self._init_output_header()
        if self.output_data is None:
            self._init_output_data()
        self._load_fir_config(pad=True)
        # self._fir_filter_coef = self._fir_filter_coef[:155]

        input_samples = self.input_data[:self._input_samples]

        input_samples_per_pol_dim = int(
            self._input_samples / (self._input_npol*self._input_ndim))
        output_samples_per_pol_dim = int(
            self._output_samples / (self._output_npol*self._input_ndim))

        if self._input_ndim == 2:
            input_samples = input_samples[:(input_samples_per_pol_dim *
                                            self._input_npol*self._input_ndim)]
            input_samples_temp = input_samples.reshape(
                (input_samples_per_pol_dim, self._input_npol*self._input_ndim))
            input_samples = np.zeros(
                (input_samples_per_pol_dim, self._input_npol),
                dtype=self._complex_dtype)

            input_samples[:, 0] = (input_samples_temp[:, 0] +
                                   1j*input_samples_temp[:, 1])
            input_samples[:, 1] = (input_samples_temp[:, 2] +
                                   1j*input_samples_temp[:, 3])

            # input_samples = input_samples[:, 0] + 1j*input_samples[:, 1]

        elif self._input_ndim == 1:
            input_samples = input_samples.reshape(
                (input_samples_per_pol_dim, self._input_npol))

        # do any downsampling necessary for conversion
        # from real to complex data.
        if int(self._ndim_ratio) != 1:
            input_samples = input_samples[::int(self._ndim_ratio), :]

        return (input_samples,
                input_samples_per_pol_dim,
                output_samples_per_pol_dim)

    def channelize(self, *args, **kwargs):
        prepped = self._prepare_channelize(*args, **kwargs)

        # if not self.oversampled:
        #     g = self._channelize_fft(*prepped)
        # else:
        g = self._channelize(*prepped)

        for i in g:
            pass
        self._dump_data(self.output_header, self.output_data)


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

    channelizer = PFBChannelizer(
        parsed.input_file_path,
        parsed.fir_file_path
    )

    channelizer.channelize(parsed.channels, parsed.oversampling_factor)
