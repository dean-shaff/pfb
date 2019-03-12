import unittest
import os
import logging

import numpy as np

from pfb.formats import DADAFile

current_dir = os.path.dirname(os.path.abspath(__file__))
test_data_dir = os.path.join(current_dir, "test_data")


class TestDADAFile(unittest.TestCase):

    def setUp(self):
        test_dada_file_paths = [os.path.join(
            test_data_dir,
            "py_channelized.impulse.noise_0.0.nseries_3.ndim_2.os.dump"
        ), os.path.join(
            test_data_dir,
            "pre_Fold.vanilla.pulsar.10.dump"
        )]
        self.dada_files = [DADAFile(f) for f in test_dada_file_paths]

    def test_getitem(self):
        self.dada_files[0]._load_data_from_file()
        self.assertTrue(self.dada_files[0]["NCHAN"] == "8")
        self.assertTrue(self.dada_files[0]["NPOL"] == "2")
        self.assertTrue(self.dada_files[0]["NDIM"] == "2")
        self.assertTrue(self.dada_files[0]["NBIT"] == "32")

        self.dada_files[1]._load_data_from_file()
        self.assertTrue(self.dada_files[1]["NCHAN"] == "1")
        self.assertTrue(self.dada_files[1]["NPOL"] == "1")
        self.assertTrue(self.dada_files[1]["NDIM"] == "4")
        self.assertTrue(self.dada_files[1]["NBIT"] == "32")

    def test_setitem(self):
        self.dada_files[0]._load_data_from_file()
        self.dada_files[0]["NCHAN"] = "10"
        self.assertTrue(self.dada_files[0]["NCHAN"] == "10")

    def test_contains(self):

        with self.assertRaises(RuntimeError):
            val = "NCHAN" in self.dada_files[0]
        self.dada_files[0]._load_data_from_file()
        self.assertTrue("NCHAN" in self.dada_files[0])

    def test_load_data_from_file(self):
        self.dada_files[0]._load_data_from_file()
        self.assertIsInstance(self.dada_files[0]._data, np.ndarray)
        self.assertTrue("NCHAN" in self.dada_files[0]._header)

    def test_shape_data(self):
        self.dada_files[0]._load_data_from_file()
        data_expected = self.dada_files[0]._shape_data(self.dada_files[0].data)
        self.assertTrue(
            all([i==j for i, j in zip(data_expected.shape, (443961, 8, 2))]))

        ndim, nchan, npol = [int(self.dada_files[0][item])
                             for item in ["NDIM", "NCHAN", "NPOL"]]

        data_flat = self.dada_files[0].data

        data_shaped_ = data_flat.reshape((-1, nchan, npol*ndim))
        data_shaped = np.zeros(
            (data_shaped_.shape[0], nchan, npol),
            dtype=self.dada_files[0]["COMPLEX_DTYPE"])

        data_shaped[:, :, 0] = data_shaped_[:, :, 0] + 1j*data_shaped_[:, :, 1]
        data_shaped[:, :, 1] = data_shaped_[:, :, 2] + 1j*data_shaped_[:, :, 3]

        self.assertTrue(np.allclose(data_expected, data_shaped))

    def test_load_data(self):
        self.dada_files[0].load_data()

    def test_dump_data(self):
        self.dada_files[0].load_data()
        new_file_path = self.dada_files[0].dump_data()
        os.remove(new_file_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
