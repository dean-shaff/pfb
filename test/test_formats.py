import unittest
import os
import logging

import numpy as np

from src.formats import DADAFile

current_dir = os.path.dirname(os.path.abspath(__file__))
test_data_dir = os.path.join(current_dir, "test_data")


class TestDADAFile(unittest.TestCase):

    def setUp(self):
        test_dada_file_path = os.path.join(
            test_data_dir,
            "py_channelized.impulse.noise_0.0.nseries_3.ndim_2.os.dump"
        )
        self.dada_file = DADAFile(test_dada_file_path)

    def test_getitem(self):
        self.dada_file._load_data_from_file()
        self.assertTrue(self.dada_file["NCHAN"] == "8")
        self.assertTrue(self.dada_file["NPOL"] == "2")
        self.assertTrue(self.dada_file["NDIM"] == "2")
        self.assertTrue(self.dada_file["NBIT"] == "32")

    def test_setitem(self):
        self.dada_file._load_data_from_file()
        self.dada_file["NCHAN"] = "10"
        self.assertTrue(self.dada_file["NCHAN"] == "10")

    def test_contains(self):

        with self.assertRaises(RuntimeError):
            val = "NCHAN" in self.dada_file
        self.dada_file._load_data_from_file()
        self.assertTrue("NCHAN" in self.dada_file)

    def test_load_data_from_file(self):
        self.dada_file._load_data_from_file()
        self.assertIsInstance(self.dada_file._data, np.ndarray)
        self.assertTrue("NCHAN" in self.dada_file._header)

    def test_shape_data(self):
        self.dada_file._load_data_from_file()
        data_expected = self.dada_file._shape_data(self.dada_file.data)
        self.assertTrue(
            all([i==j for i, j in zip(data_expected.shape, (443961, 8, 2))]))

        ndim, nchan, npol = [int(self.dada_file[item])
                             for item in ["NDIM", "NCHAN", "NPOL"]]

        data_flat = self.dada_file.data

        data_shaped_ = data_flat.reshape((-1, nchan, npol*ndim))
        data_shaped = np.zeros(
            (data_shaped_.shape[0], nchan, npol),
            dtype=np.complex64)

        data_shaped[:, :, 0] = data_shaped_[:, :, 0] + 1j*data_shaped_[:, :, 1]
        data_shaped[:, :, 1] = data_shaped_[:, :, 2] + 1j*data_shaped_[:, :, 3]

        self.assertTrue(np.allclose(data_expected, data_shaped))

    def test_load_data(self):
        self.dada_file.load_data()

    def test_dump_data(self):
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
