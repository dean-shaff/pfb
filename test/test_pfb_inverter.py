import unittest
import logging
import os

import numpy as np
import matplotlib.pyplot as plt

from src.pfb.pfb_inverter import PFBInverter

current_dir = os.path.dirname(__file__)
test_dir = os.path.join(current_dir, "test_data")

os_data_file = os.path.join(
    test_dir,
    "py_channelized.impulse.noise_0.0.nseries_3.ndim_2.os.dump")
cs_data_file = os.path.join(
    test_dir,
    "py_channelized.impulse.noise_0.0.nseries_3.ndim_2.cs.dump")


class TestPFBInverter(unittest.TestCase):

    def test_critically_sampled_pfb_inverter(self):
        filter_size = 328
        inverter = PFBInverter(os_data_file)
        inverter._load_input_data()
        g = inverter._invert(inverter.input_data)
        input_time_series = next(g)
        output_fft_size = next(g)
        ndat, nchan, npol = input_time_series.shape
        fig, axes = plt.subplots(nchan)
        # fig.tight_layout()
        for i in range(nchan):
            axes[i].set_xlim([0, 200])
            axes[i].grid(True)
            axes[i].plot(np.real(input_time_series[:, i, 0]))
            axes[i].plot(np.imag(input_time_series[:, i, 0]))
        # plt.show()
        for i in range(2):
            for p in range(3):
                assembled = next(g)
                assembled_rolled = next(g)
                next(g)
        fig, axes = plt.subplots(2, 1)
        for i in range(axes.shape[0]):
            axes[i].grid(True)
        axes[0].plot(np.log10(np.abs(assembled_rolled)))
        axes[1].plot(np.angle(assembled))

        for dat in g:
            pass
        ndat, npol = dat.shape
        fig, axes = plt.subplots(2, 1)
        xlim = [0, 500]
        for i in range(axes.shape[0]):
            axes[0].set_xlim(xlim)
            axes[i].grid(True)
        axes[0].plot(np.real(dat[:, 0]))
        axes[0].plot(np.imag(dat[:, 0]))
        axes[0].axvline(filter_size)
        axes[1].plot(np.log10(np.abs(np.fft.fft(dat[2*output_fft_size:3*output_fft_size, 0]))))
        plt.show()



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
