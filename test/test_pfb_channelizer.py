import os
import sys
import logging
import unittest

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "test_data")

sys.path.append(os.path.dirname(current_dir))

from src.pfb_channelizer import PFBChannelizer
from src.util import load_dada_file

fir_file_path = os.path.join(
    current_dir, "OS_Prototype_FIR_8.mat")
input_file_path = os.path.join(
    data_dir, "impulse.noise_0.0.nseries_3.ndim_2.dump")


def compare_dump_files(file_path0, file_path1, **kwargs):
    print(file_path0)
    print(file_path1)
    comp_dat = []
    dat_sizes = np.zeros(2)
    fnames = [file_path0, file_path1]
    for i, fname in enumerate(fnames):
        header, data = load_dada_file(fname)
        comp_dat.append(data)
        dat_sizes[i] = data.shape[0]
    min_size = int(np.amin(dat_sizes))
    comp_dat = [d[:min_size] for d in comp_dat]

    # fig, axes = plt.subplots(3, 1)
    # axes[0].plot(comp_dat[0])
    # axes[1].plot(comp_dat[1])
    # axes[2].plot(np.abs(np.subtract(*comp_dat)))
    # plt.show()

    return np.allclose(*comp_dat, **kwargs)


class TestPFBChannelizer(unittest.TestCase):

    def setUp(self):
        channelizer = PFBChannelizer(
            input_file_path,
            fir_file_path
        )
        self.channelizer = channelizer

    @unittest.skip("")
    def test_new_filter(self):

        g = self.channelizer._channelize(8, "1/1")
        next(g)
        f, lf = next(g)
        c = 0
        for j in range(8):
            print(f"j={j}: {np.allclose(f[:, j], lf[:, j], atol=1e-5)}")
        print(np.allclose(f, lf, atol=1e-5))
        fig, axes = plt.subplots(3, 1)
        xlim = [0, 100]
        for ax in axes[:2]:
            ax.grid(True)
            ax.set_xlim(xlim)
        axes[0].plot(f[:, c].real)
        axes[0].plot(f[:, c].imag)
        axes[1].plot(lf[:, c].real)
        axes[1].plot(lf[:, c].imag)
        axes[2].plot(np.abs(f[:, c] - lf[:, c]).real)
        plt.show()

    # @unittest.skip("")
    def test_critically_sampled_pfb_vs_matlab(self):

        expected_file_path = os.path.join(
            data_dir,
            "full_channelized_impulse.noise_0.0.nseries_3.ndim_2.cs.dump"
        )
        self.channelizer.channelize(8, "1/1")

        self.assertTrue(
            compare_dump_files(expected_file_path,
                               self.channelizer.output_file_path, atol=1e-5))

    # @unittest.skip("")
    def test_over_sampled_pfb_vs_matlab(self):

        expected_file_path = os.path.join(
            data_dir,
            "full_channelized_impulse.noise_0.0.nseries_3.ndim_2.os.dump"
        )

        self.channelizer.channelize(8, "8/7")
        self.assertTrue(
            compare_dump_files(expected_file_path,
                               self.channelizer.output_file_path, atol=1e-5))


if __name__ == '__main__' :
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
