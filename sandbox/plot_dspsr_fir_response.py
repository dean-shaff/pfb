import os

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from src.pfb import util

dspsr_dump = "/home/SWIN/dshaff/ska/dspsr-sf/freq_response.dat"
cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(cur_dir)
test_dir = os.path.join(parent_dir, "test")

fir_coeff_file_path = os.path.join(test_dir, "Prototype_FIR.mat")


def main():
    _, fir_coeff = util.load_matlab_filter_coef(fir_coeff_file_path)

    w, h = scipy.signal.freqz(fir_coeff, 1, worN=1024)
    data = np.zeros((h.shape[0], 2), dtype=np.float32)
    data[:, 0] = h.real
    data[:, 1] = h.imag
    data = data.flatten()

    with open(dspsr_dump, "rb") as f:
        buffer = f.read()
        dspsr_data = np.frombuffer(
            buffer, dtype=np.dtype(np.float32)
        )

    fig, axes = plt.subplots(2, 1)

    axes[0].plot(data[::2])
    axes[0].plot(data[1::2])
    axes[1].plot(dspsr_data[::2])
    axes[1].plot(dspsr_data[1::2])

    plt.show()


if __name__ == "__main__":
    main()
