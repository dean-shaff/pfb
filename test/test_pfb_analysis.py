import logging
import unittest

import numpy as np

from pfb.pfb_analysis import (
    pfb_analysis,
    calc_output_tsamp)


class TestPFBAnalysis(unittest.TestCase):

    def test_pfb_analysis(self):
        analyzer = pfb_analysis(os_factor="8/7",
                                nchan=8,
                                fir_filter_coeff=np.random.rand(88))
        ndat = int(1e6)
        sample_data = np.random.rand(ndat) + 1j*np.random.rand(ndat)
        analyzer(sample_data)

    def test_calc_output_tsamp(self):
        input_tsamp = 0.025
        output_tsamp = calc_output_tsamp(input_tsamp,
                                         nchan=8,
                                         os_factor="8/7")
        self.assertTrue(np.allclose(output_tsamp, 0.175))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
