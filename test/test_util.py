import os
import unittest

import numpy as np

from src.pfb import util

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "test_data")
fir_file_path = os.path.join(
    current_dir, "Prototype_FIR.mat")


class TestDumpFilterCoef(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.coef_config, cls.coef = util.load_matlab_filter_coef(fir_file_path)

    def test_dump_filter_coef(self):
        coef_str = util.dump_filter_coef(self.coef)
        self.assertTrue(len(coef_str) == len(self.coef))


class TestLoadMatlabFilterCoef(unittest.TestCase):

    def test_load_matlab_filter_coef(self):
        coef_config, coef = util.load_matlab_filter_coef(fir_file_path)


class TestAddFilterInfoToHeader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ntaps = 341
        cls.coef = np.random.rand(cls.ntaps)

    def test_add_filter_info_to_header_one_level(self):
        header = {}
        filter_info = [{
            "COEFF": self.coef,
            "OVERSAMP": "8/7",
            "NTAP": len(self.coef),
            "NCHAN_PFB": "8"
        }]
        header_updated = util.add_filter_info_to_header(header, filter_info)
        self.assertTrue(header_updated["NSTAGE"] == 1)

    def test_add_filter_info_to_header_n_level(self):
        n = 3
        header = {}
        filter_info = [{
            "COEFF": self.coef,
            "OVERSAMP": "8/7",
            "NTAP": len(self.coef),
            "NCHAN_PFB": "8"
        } for i in range(n)]
        header_updated = util.add_filter_info_to_header(header, filter_info)
        self.assertTrue(header_updated["NSTAGE"] == n)


class TestDumpDadaFile(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_file_path = os.path.join(
            data_dir, "test.dada")
        cls.test_file_path_no_coeff = os.path.join(
            data_dir, "test.no_coeff.dada")
        cls.data = np.zeros(100)
        cls.ntaps = 341
        cls.coef = np.random.rand(cls.ntaps)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_dump_dada_file_no_filter_coef(self):
        header = {"HDR_SIZE": 4096}
        util.dump_dada_file(
            self.test_file_path_no_coeff,
            header,
            self.data
        )

    def test_dump_dada_file(self):
        filter_info = [{
            "COEFF": self.coef,
            "OVERSAMP": "8/7",
            "NTAP": len(self.coef),
            "NCHAN_PFB": "8"
        }]
        header = {"HDR_SIZE": 4096}
        header = util.add_filter_info_to_header(
            header, filter_info)
        util.dump_dada_file(
            self.test_file_path,
            header,
            self.data
        )


if __name__ == '__main__':
    unittest.main()
