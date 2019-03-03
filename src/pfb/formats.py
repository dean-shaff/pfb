import os
import logging

import numpy as np

from .util import (
    load_dada_file,
    dump_dada_file
)

module_logger = logging.getLogger(__name__)

__all__ = [
    "DataFile",
    "DADAFile"
]


class DataFile:

    def __init__(self, file_path):
        self.logger = module_logger.getChild("DataFile")
        self._file_path = file_path
        self._header = None
        self._data = None

    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, file_path):
        self._file_path = file_path
        self._data = None
        self._header = None

    @property
    def header(self):
        return self._header

    @header.setter
    def header(self, new_header):
        self._header = new_header

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        # if new_data.ndim != 1 and new_data.ndim != 4:
        #     raise RuntimeError(f"Ambiguous data shape: {new_data.ndim}")
        iscomplex = np.iscomplex(new_data)
        ndim = 1
        if np.all(iscomplex):
            ndim = 2
        ndat = new_data.shape[0]

        if new_data.ndim == 1:
            nchan, npol = 1, 1
            self._data = np.zeros((ndat, nchan, npol), dtype=new_data.dtype)
            self._data[:, 0, 0] = new_data
        else:
            nchan, npol = new_data.shape[1], new_data.shape[2]
            self._data = new_data

        self["NDAT"], self["NCHAN"], self["NPOL"], self["NDIM"] = \
            [str(i) for i in [ndat, nchan, npol, ndim]]

    @property
    def nchan(self):
        return int(self["NCHAN"])

    @property
    def ndim(self):
        return int(self["NDIM"])

    @property
    def npol(self):
        return int(self["NPOL"])

    @property
    def ndat(self):
        if self.data is not None:
            if self.data.ndim > 1:
                return self.data.shape[0]

    def __getitem__(self, item):
        if self._header is not None:
            if item in self._header:
                return self._header[item]

    def __setitem__(self, item, val):
        if self._header is not None:
            if item in self._header:
                self._header[item] = val

    def __contains__(self, item):
        if self._header is not None:
            if item in self._header:
                return True
            else:
                return False
        else:
            raise RuntimeError(("DataFile.__contains__: Need to load "
                                "data from file before calling __contains__"))

    def load_data(self):
        raise NotImplemented()

    def dump_data(self, overwrite=False):
        raise NotImplemented()


class DADAFile(DataFile):

    default_header = {
        "HDR_VERSION": "1.0",
        "HDR_SIZE": "4096",
        "TELESCOPE": "PKS",
        "PRIMARY": "dspsr",
        "UTC_START": "2007-05-18-15:55:58",
        "SOURCE": "J1644-4559",
        "RA": "16:44:49.28",
        "DEC": "-45:59:09.5",
        "FREQ": "1405.000000",
        "BW": "40",
        "TSAMP": "0.0125",
        "NBIT": "32",
        "NDIM": "2",
        "NPOL": "2",
        "NCHAN": "1",
        "MODE": "PSR",
        "OBS_OFFSET": "0",
        "INSTRUMENT": "dspsr",
        "DSB": "0"
    }

    def __init__(self, file_path):
        super(DADAFile, self).__init__(file_path)
        self._header = self.default_header.copy()
        self.logger = module_logger.getChild("DADAFile")

    def _load_data_from_file(self):

        self._header, self._data = load_dada_file(self.file_path)

    def _shape_data(self, data):

        if self._header is None:
            raise RuntimeError(("DADAFile._shape_data: Need to load "
                                "data from file before calling _shape_data"))

        ndim, nchan, npol = [int(self[item])
                             for item in ["NDIM", "NCHAN", "NPOL"]]

        data = data.reshape((-1, nchan, npol, ndim))
        if ndim == 2:  # means we're dealing with complex data
            data = data[:, :, :, 0] + 1j*data[:, :, :, 1]

        return data

    def load_data(self):

        self._load_data_from_file()
        self._data = self._shape_data(self._data).copy()

    def dump_data(self, overwrite=True):

        new_file_path = self.file_path
        if not overwrite:
            exists = os.path.exists(new_file_path)
            temp_file_path = new_file_path
            # temp_file_path = f"{new_file_path}.{i}"
            i = 0
            while exists:
                temp_file_path = f"{new_file_path}.{i}"
                exists = os.path.exists(temp_file_path)
                i += 1
            new_file_path = temp_file_path

        if self.ndim == 1:
            data = self.data.flatten()
        else:
            data = np.zeros((self.ndat, self.nchan, self.ndim*self.npol),
                            dtype=np.float32)
            for pol in range(self.npol):
                data[:, :, pol*2] = self.data[:, :, pol].real
                data[:, :, pol*2 + 1] = self.data[:, :, pol].imag
        self.logger.debug(f"dump_data: new file path: {new_file_path}")
        dump_dada_file(new_file_path, self.header, data)
        return new_file_path
