import logging
import time
import os

import numpy as np
import scipy.io

module_logger = logging.getLogger(__name__)

__all__ = [
    "load_dada_file",
    "dump_dada_file",
    "load_matlab_filter_coef",
    "get_most_recent_data_file"
]

_dtype_map = {
    '32': np.float32,
    '64': np.float64
}


def _process_header(header_arr):
    header_str = "".join([c.decode("UTF-8") for c in header_arr.tolist()])
    lines = header_str.split("\n")
    header = {}
    for line in lines:
        if line.startswith("#") or not line:
            continue
        else:
            key, val = line.split()[:2]
            header[key] = val
    return header


def load_dada_file(file_path, header_size=4096):
    with open(file_path, "rb") as file:
        buffer = file.read()
        header = np.frombuffer(
            buffer, dtype='c', count=header_size
        )
        header = _process_header(header)
        float_dtype = _dtype_map[str(header["NBIT"])]
        data = np.frombuffer(
            buffer, dtype=float_dtype, offset=header_size
        )
    return [header, data]


def dump_dada_file(file_path, header, data, header_size=4096):
    module_logger.debug(f"dump_dada_file file_path: {file_path}")
    module_logger.debug(f"dump_dada_file header: {header}")
    module_logger.debug(f"dump_dada_file header_size: {header_size}")
    t0 = time.time()

    header_str = "\n".join(
        [f"{key} {header[key]}" for key in header]) + "\n"
    header_bytes = str.encode(header_str)
    remaining_bytes = header_size - len(header_bytes)
    module_logger.debug(
        f"dump_dada_file len(header_bytes): {len(header_bytes)}")
    module_logger.debug(
        f"dump_dada_file remaining_bytes: {remaining_bytes}")
    header_bytes += str.encode(
        "".join(["\0" for i in range(remaining_bytes)]))

    assert len(header_bytes) == header_size, \
        f"Number of bytes in header must be equal to {header_size}"

    with open(file_path, "wb") as output_file:
        output_file.write(header_bytes)
        output_file.write(data.flatten().tobytes())

    module_logger.debug(
        f"dump_dada_file Took {time.time() - t0:.4f} seconds to dump data")


def load_matlab_filter_coef(file_path):
    fir_config = scipy.io.loadmat(file_path)
    fir_filter_coef = fir_config["h"].reshape(-1)
    return fir_config, fir_filter_coef


def get_most_recent_data_file(directory,
                              prefix="simulated_pulsar", suffix=".dump"):

    file_paths = []
    for fname in os.listdir(directory):
        if fname.startswith(prefix):
            fpath = os.path.join(directory, fname)
            file_paths.append(
                (fpath,
                 os.path.getmtime(fpath))
            )

    file_paths.sort(key=lambda x: x[1])

    return file_paths[-1][0]
