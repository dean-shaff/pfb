import typing
import functools
import os

import numpy as np
import psr_formats

from . import util
from .pfb_analysis import pfb_analysis, calc_output_tsamp
from .rational import Rational

fir_filter_coeff_type = typing.Union[str, np.ndarray]
os_factor_type = typing.Union[str, Rational]


class FormatHandler:

    def __call__(self, input_file: typing.Any, **kwargs):
        self.psr_format_cls = type(input_file)
        input_data = self.load_input_data(input_file)
        output_file = self.prepare_output_data(input_file, **kwargs)
        output_data = self.apply(input_data)
        self.dump_output_data(output_file, output_data)
        return output_file

    def load_input_data(self, input_file) -> np.ndarray:
        raise NotImplementedError()

    def apply(self, input_data) -> np.ndarray:
        raise NotImplementedError()

    def prepare_output_data(self,
                            input_file: typing.Any,
                            *,
                            output_dir: str,
                            output_file_name: str) -> typing.Any:
        raise NotImplementedError()

    def dump_output_data(self, output_file, output_data) -> None:
        raise NotImplementedError()


class PSRFormatHandler(FormatHandler):

    def load_input_data(self, input_file: psr_formats.DataFile) -> np.ndarray:
        if not input_file.loaded:
            input_file.load_data()
        return input_file.data

    def dump_output_data(self,
                         output_file: psr_formats.DataFile,
                         output_data: np.ndarray):
        output_file.data = output_data
        output_file.dump_data()


class PSRFormatChannelizer(PSRFormatHandler):

    def __init__(self,
                 *,
                 os_factor: os_factor_type,
                 nchan: int,
                 fir_filter_coeff: fir_filter_coeff_type):
        super(PSRFormatChannelizer, self).__init__()

        self._os_factor = Rational.from_str(os_factor)
        self.nchan = nchan
        if hasattr(fir_filter_coeff, "format"):
            fir_filter_coeff = util.load_matlab_filter_coeff(
                fir_filter_coeff)[1]
        self._fir_filter_coeff = fir_filter_coeff

    @property
    def fir_filter_coeff(self):
        return self._fir_filter_coeff

    @fir_filter_coeff.setter
    def fir_filter_coeff(self,
                         fir_filter_coeff: fir_filter_coeff_type):
        if hasattr(fir_filter_coeff, "format"):
            fir_filter_coeff = util.load_matlab_filter_coeff(
                fir_filter_coeff)[1]
        self._fir_filter_coeff = fir_filter_coeff

    @property
    def os_factor(self):
        return self._os_factor

    @os_factor.setter
    def os_factor(self, os_factor: os_factor_type):
        self._os_factor = Rational.from_str(os_factor)

    def prepare_output_data(self,
                            input_file: typing.Any,
                            *,
                            output_dir: str,
                            output_file_name: str):
        psr_format_cls = type(input_file)
        output_file_path = os.path.join(output_dir, output_file_name)
        output_file = psr_format_cls(output_file_path)
        output_file["TSAMP"] = calc_output_tsamp(
            float(input_file["TSAMP"]),
            input_ndim=util.dtype_to_int[input_file.data.dtype],
            output_ndim=2,
            nchan=self.nchan,
            os_factor=self.os_factor
        )

        output_file.header["OS_FACTOR"] = str(self.os_factor)
        output_file.header["PFB_DC_CHAN"] = 1
        # add filter info to the output_file
        fir_info = [{
            "OVERSAMP": str(self.os_factor),
            "COEFF": self.fir_filter_coeff,
            "NCHAN_PFB": self.nchan
        }]
        output_file.header.update(util.filter_info_to_dict(fir_info))
        return output_file

    def apply(self, input_data: np.ndarray):
        channelizer = pfb_analysis(
            fir_filter_coeff=self.fir_filter_coeff,
            os_factor=self.os_factor,
            nchan=self.nchan
        )

        expander = functools.partial(np.expand_dims, axis=2)

        n_pol = input_data.shape[-1]
        output_data = expander(channelizer(input_data[:, 0, 0]))
        for i_pol in range(1, n_pol):
            output_ipol = expander(channelizer(input_data[:, 0, i_pol]))
            output_data = np.concatenate((output_data, output_ipol), axis=2)
        return output_data