import os

import numpy as np
from psr_formats import DADAFile

from pfb import util, pfb_synthesis, rational

cur_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(cur_dir)
test_dir = os.path.join(parent_dir, "test", "test_data")
output_dir = "/home/SWIN/dshaff/ska/test_data"

dada_file_name = "fir.{}.dada"

fir_coeff_file_path = os.path.join(test_dir, "Prototype_FIR.4-3.8.80.mat")
os_factor = "4/3"
nchan = 8
N = 128


def main():
    _, fir_coeff = util.load_matlab_filter_coeff(fir_coeff_file_path)

    N_normalized = rational.Rational.from_str(os_factor).normalize(N)

    deripple_response = pfb_synthesis._multi_channel_deripple_response(
        nchan, N_normalized, fir_coeff, pfb_dc_chan=True, dtype=np.float32
    )


    data = np.zeros((deripple_response.shape[0], 1, 1), dtype=np.complex64)
    data[:, 0, 0] = (
        deripple_response +
        1j*np.zeros(deripple_response.shape[0], dtype=np.float32))

    dada_file_path = os.path.join(
        output_dir, dada_file_name.format(data.shape[0]))

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
        "OBS_OFFSET": "0",
        "OS_FACTOR": os_factor,
        "PFB_DC_CHAN": "1"
    }

    fir_info = [{
        "OVERSAMP": os_factor,
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
