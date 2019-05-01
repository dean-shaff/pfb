import logging
import time

import numpy as np

from .rational import Rational

module_logger = logging.getLogger(__name__)


class PFBInverter:

    def __init__(self, input_file_path):
        self.logger = module_logger.getChild("PFBInverter")
        self.input_file_path = input_file_path

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

    def _load_input_data(self):
        t0 = time.time()
        header, data = load_dada_file(self.input_file_path)

        self.logger.debug(
            (f"_load_input_data: "
             f"Took {time.time() - t0:.3f} seconds to load input data")
        )

        self._float_dtype = data.dtype
        self._complex_dtype = (np.complex64 if
                               self._float_dtype == np.float32
                               else np.complex128)

        self.input_header = header

        self._input_npol = int(self.input_header["NPOL"])
        self._input_ndim = int(self.input_header["NDIM"])
        self._input_nchan = int(self.input_header["NCHAN"])
        self.oversampling_factor = Rational(*self.input_header["OS_FACTOR"].split('/'))

        data = data.reshape(
            (-1, self._input_nchan, self._input_npol*self._input_ndim))

        self.input_data = np.zeros(
            (data.shape[0], self._input_nchan, self._input_npol),
            dtype=self._complex_dtype
        )

        self.input_data[:, :, 0] = data[:, :, 0] + 1j*data[:, :, 1]
        self.input_data[:, :, 1] = data[:, :, 2] + 1j*data[:, :, 3]

    def _invert(self, input_time_series,
                input_fft_size=32768, output_fft_size=None):
        offset = 0
        input_time_series = input_time_series[offset:, :, :]
        yield input_time_series
        if output_fft_size is None:
            output_fft_size = self._input_nchan *\
                              self.oversampling_factor.normalize(
                                input_fft_size)
        yield output_fft_size

        os_keep = self.oversampling_factor.normalize(input_fft_size)
        os_keep_2 = int(os_keep // 2)
        os_discard = input_fft_size - os_keep
        os_discard_2 = int(os_discard // 2)

        n_parts = int(input_time_series.shape[0] // input_fft_size)

        self.logger.debug(
            ("_invert"
             f"input_fft_size: {input_fft_size} "
             f"output_fft_size: {output_fft_size} "
             f"oversampling keep region: {os_keep} "
             f"n_part: {n_parts}")
        )

        assembled = np.zeros(output_fft_size, dtype=self._complex_dtype)
        output_data = np.zeros((output_fft_size*n_parts, self._input_npol), dtype=self._complex_dtype)

        for ipol in range(self._input_npol):
            for ipart in range(n_parts):
                in_step = ipart*input_fft_size
                out_step = ipart*output_fft_size
                for ichan in range(self._input_nchan):
                    time_domain_ichan = \
                        input_time_series[in_step: in_step+input_fft_size, ichan, ipol]
                    freq_domain_ichan = \
                        np.fft.fft(time_domain_ichan)

                    idx = ichan*os_keep

                    assembled[idx: idx+os_keep_2] = \
                        freq_domain_ichan[input_fft_size-os_keep_2:]

                    assembled[idx+os_keep_2: idx+os_keep] = \
                        freq_domain_ichan[os_discard_2:int(input_fft_size/2)]
                yield assembled
                assembled = np.roll(assembled, -int(os_keep/2))
                yield assembled
                output_data[out_step:out_step + output_fft_size, ipol] = \
                    np.fft.ifft(assembled)
                yield output_data[out_step:out_step + output_fft_size, ipol]
        output_data /= float(self.oversampling_factor)
        yield output_data

    def _init_output_data(self):
        pass

    def invert(self, **kwargs):
        if self.input_data is None:
            self._load_input_data()
        self._init_output_data()

        g = self._invert(self.input_data, **kwargs)
        for i in g:
            pass
