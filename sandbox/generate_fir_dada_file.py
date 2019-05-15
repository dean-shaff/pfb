import os

import numpy as np
import scipy.signal
from psr_formats import DADAFile

from pfb import util

cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(cur_dir)
test_dir = os.path.join(parent_dir, "test", "test_data")

dada_file_name = "fir.{}.dada"

fir_coeff_file_path = os.path.join(test_dir, "Prototype_FIR.4-3.8.80.mat")


def freqz(a, N):
    f = np.fft.rfft(a, 2*N)
    return f[:N]


def main():
    N = 1024
    _, fir_coeff = util.load_matlab_filter_coeff(fir_coeff_file_path)

    w, h = scipy.signal.freqz(fir_coeff, 1, worN=N)
    # h_ = freqz(fir_coeff, N)

    data = np.zeros((h.shape[0], 1, 2), dtype=np.float32)
    data[:, 0, 0] = h.real
    data[:, 0, 1] = h.imag

    # for i in range(100):
    #     print(data[i, :])

    dada_file_path = os.path.join(
        parent_dir, dada_file_name.format(data.shape[0]))

    header = {
        "NCHAN": 1,
        "NBIT": 32,
        "FLOAT_DTYPE": np.float32,
        "NDIM": 2,
        "NPOL": 1,
        "BW": 40,
        "FREQ": 1405,
        "HDR_VERSION": 1.0,
        "INSTRUMENT": "dspsr",
        "HDR_SIZE": 4096,
        "TELESCOPE": "PKS",
        "SOURCE": "J1644-4559",
        "TSAMP": 0.175,
        "UTC_START": "2019-02-06-04:57:23",
        "OBS_OFFSET": "0"
    }

    fir_info = [{
        "OVERSAMP": "1/1",
        "NTAP": len(fir_coeff),
        "COEFF": fir_coeff,
        "NCHAN_PFB": 1
    }]
    output_file = DADAFile(dada_file_path)
    output_file.header = header
    output_file.header.update(util.filter_info_to_dict(fir_info))
    output_file.data = data
    output_file.dump_data()


if __name__ == "__main__":
    main()
