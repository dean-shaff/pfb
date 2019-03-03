import os
import logging
import unittest

import numpy as np
import matplotlib.pyplot as plt

from src.pfb.pfb_channelizer import PFBChannelizer
from src.pfb.util import load_dada_file

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "test_data")


# This is the FIR filter file for the matlab channelized data.
fir_file_path = os.path.join(
    current_dir, "OS_Prototype_FIR_8.matlab.mat")
# fir_file_path = os.path.join(
#     current_dir, "Prototype_FIR.mat")
# fir_file_path = os.path.join(
#     current_dir, "Prototype_FIR.120.mat")

input_file_path = os.path.join(
    data_dir, "impulse.noise_0.0.nseries_3.ndim_2.dump")
# input_file_path = os.path.join(
#     data_dir, "simulated_pulsar.noise_0.0.nseries_3.ndim_2.dump")


def compare_dump_files(file_path0, file_path1, **kwargs):
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
        channelizer = PFBChannelizer.from_input_files(
            input_file_path,
            fir_file_path
        )
        self.channelizer = channelizer

    @unittest.skip("")
    def test_filter_response(self):
        nchan = 8
        samples = self.channelizer._prepare_channelize(nchan, "1/1")
        g = self.channelizer._channelize(samples)
        filtered = next(g)
        fft_size = filtered.shape[0]

        fig, axes = plt.subplots(3, nchan, figsize=(18, 10))
        fig.tight_layout(rect=[0.05, 0.03, 1, 0.95])
        # xlim = [2, 18]
        xlim = [0, 50]
        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                axes[i, j].grid(True)
                axes[0, j].set_xlim(xlim)
        filter_coef_per_chan = self.channelizer.fir_filter_coeff.shape[0] // nchan
        for c in range(nchan):
            axes[0, c].plot(filtered[:, c].real)
            axes[0, c].plot(filtered[:, c].imag)
            chan_fft = np.fft.fft(filtered[:, c][filter_coef_per_chan:fft_size+filter_coef_per_chan])
            axes[1, c].plot(np.abs(chan_fft))
            axes[2, c].plot(np.angle(chan_fft))

        plt.show()

    @unittest.skip("")
    def test_new_filter(self):
        nchan = 8
        prepped = self.channelizer._prepare_channelize(nchan, "8/7")
        g0 = self.channelizer._channelize(*prepped)
        g1 = self.channelizer._channelize_fft(*prepped)
        f0, f1 = next(g0), next(g1)
        c = 4
        for j in range(nchan):
            print(f"j={j}: {np.allclose(f0[:, j], f1[:, j], atol=1e-5)}")
        print(np.allclose(f0, f1, atol=1e-5))
        fig, axes = plt.subplots(3, nchan, figsize=(18, 10))
        fig.tight_layout(rect=[0.05, 0.03, 1, 0.95])
        # xlim = [2, 18]
        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                axes[i, j].grid(True)
                # axes[i, j].set_xlim(xlim)

        for c in range(nchan):
            axes[0, c].plot(f0[:, c].real)
            axes[0, c].plot(f0[:, c].imag)
            axes[1, c].plot(f1[:, c].real)
            axes[1, c].plot(f1[:, c].imag)
            axes[2, c].plot(np.abs(f0[:, c] - f1[:, c]))
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


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
