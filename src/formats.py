import logging

import numpy as np

from .util import (
    load_dada_file,
    dump_dada_file
)

module_logger = logging.getLogger(__name__)

__all__ = ["DADAFile"]


class DADAFile:

    def __init__(self, file_path):
        self.logger = module_logger.getChild("DADAFile")
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

    @property
    def data(self):
        return self._data

    @property
    def nchan(self)::
        return int(self["NCHAN"])

    @property
    def ndim(self)::
        return int(self["NDIM"])

    @property
    def npol(self)::
        return int(self["NPOL"])

    @property
    def ndat(self):
        if self.data is not None:
            if self.data.ndim > 1:
                return self.data.shape[0]

    def _load_data_from_file(self):

        self._header, self._data = load_dada_file(self.file_path)

    def _shape_data(self, data):

        if self._header is None:
            raise RuntimeError(("DADAFile._shape_data: Need to load "
                                "data from file before calling _shape_data"))

        ndim, nchan, npol = [int(self[item])
                             for item in ["NDIM", "NCHAN", "NPOL"]]

        data = data.reshape((-1, nchan, npol, ndim))
        if ndim == 2:
            data = data[:, :, :, 0] + 1j*data[:, :, :, 1]

        return data

    def load_data(self):

        self._load_data_from_file()
        self._data = self._shape_data(self._data)


    def dump_data(self, overwrite=False):

        new_file_path = self.file_path
        if not overwrite:
            exists = True
            i = 1
            temp_file_path = f"{new_file_path}.{i}"
            exists = os.path.exists(temp_file_path)
            while exists:
                i += 1
                temp_file_path = f"{new_file_path}.{i}"
                exists = os.path.exists(temp_file_path)
            new_file_path = temp_file_path

        if self.ndim = 1:
            data = self.data.flatten()
        else:
            data = np.zeros((self.ndat, self.nchan, self.ndim*self.npol))
            for pol in range(self.npol):
                data[:, :, pol*2] = self.data[:,:,pol].real
                data[:, :, pol*2 + 1] = self.data[:,:,pol].imag

        dump_dada_file(new_file_path, self.header, data)

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
            raise RuntimeError(("DADAFile.__contains__: Need to load "
                                "data from file before calling __contains__"))
