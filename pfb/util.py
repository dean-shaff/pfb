import argparse
import logging
import os
import typing

import numpy as np
import scipy.io

module_logger = logging.getLogger(__name__)

__all__ = [
    "load_dada_file",
    "dump_dada_file",
    "load_matlab_filter_coef",
    "dump_filter_coef",
    "get_most_recent_data_file"
]

_float_dtype_map = {
    '32': np.float32,
    '64': np.float64
}

_complex_dtype_map = {
    '32': np.complex64,
    '64': np.complex128
}

_exclude_header_keys = ["COMPLEX_DTYPE", "FLOAT_DTYPE"]


def _process_header(header_arr: np.ndarray) -> dict:
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


def load_dada_file(file_path: str) -> typing.List:
    header_size = 4096  # smalled header size supported by DADA format
    header_read = False
    with open(file_path, "rb") as file:
        buffer = file.read()

    while not header_read:
        header = np.frombuffer(
            buffer, dtype='c', count=header_size
        )
        header = _process_header(header)
        if "HDR_SIZE" in header:
            new_header_size = int(header["HDR_SIZE"])
            if new_header_size == header_size:
                header_read = True
            else:
                header_size = new_header_size
        else:
            header_size *= 2

    float_dtype = _float_dtype_map[str(header["NBIT"])]
    complex_dtype = _complex_dtype_map[str(header["NBIT"])]
    header["FLOAT_DTYPE"] = float_dtype
    header["COMPLEX_DTYPE"] = complex_dtype
    data = np.frombuffer(
        buffer, dtype=float_dtype, offset=header_size
    )
    return [header, data]


def add_filter_info_to_header(
    header: dict,
    filter_info: typing.List[dict]
) -> dict:
    nstage = len(filter_info)
    header["NSTAGE"] = nstage
    for i in range(nstage):
        filter_coef = filter_info[i]["COEFF"]
        filter_coef_str = ",".join(
            dump_filter_coef(filter_coef))

        header[f"OVERSAMP_{i}"] = filter_info[i]["OVERSAMP"]
        header[f"NTAP_{i}"] = len(filter_coef)
        header[f"COEFF_{i}"] = filter_coef_str
        header[f"NCHAN_PFB_{i}"] = filter_info[i]["NCHAN_PFB"]
    return header


def dump_dada_file(file_path: str,
                   header: dict,
                   data: np.ndarray) -> None:
    module_logger.debug(f"dump_dada_file file_path: {file_path}")
    module_logger.debug(f"dump_dada_file header: {header}")

    def header_to_str(header: dict) -> str:
        header_str = "\n".join(
            [f"{key} {header[key]}" for key in header
             if key not in _exclude_header_keys]) + "\n"
        return header_str

    header_size = int(header["HDR_SIZE"])
    header_str = header_to_str(header)
    header_len = len(header_str)
    while header_size < header_len:
        header_size *= 2
        header["HDR_SIZE"] = header_size
        header_str = header_to_str(header)
        header_len = len(header_str)

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

    # module_logger.debug(
    #     f"dump_dada_file Took {time.time() - t0:.4f} seconds to dump data")


def load_matlab_filter_coef(file_path: str) -> typing.Tuple:
    fir_config = scipy.io.loadmat(file_path)
    fir_filter_coef = fir_config["h"].reshape(-1)
    return fir_config, fir_filter_coef


def dump_filter_coef(filter_coef: np.ndarray) -> typing.List[str]:
    """
    Given some filter coefficients, dump them to ascii format.

    Returns:
        list: a list of strings
    """
    filter_coef_as_ascii = ["{:.6E}".format(n) for n in filter_coef]
    return filter_coef_as_ascii


def get_most_recent_data_file(directory: str,
                              prefix: str = "simulated_pulsar",
                              suffix: str = ".dump") -> str:

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


def _add_fir_data_to_existing_file(
    file_path: str,
    fir_file_path: str,
    os_factor: str,
    channels: int,
    overwrite: bool = False
) -> None:
    _, coeff = load_matlab_filter_coef(fir_file_path)

    fir_info = [{
        "COEFF": coeff,
        "NTAPS": len(coeff),
        "OVERSAMP": str(os_factor),
        "NCHAN_PFB": channels
    }]

    header, data = load_dada_file(file_path)
    header = add_filter_info_to_header(header, fir_info)
    output_file_path = file_path
    counter = 0
    if not overwrite:
        output_file_path = f"{output_file_path}.{counter}"
        while os.path.exists(output_file_path):
            counter += 1
            output_file_path_split = output_file_path.split(".")
            output_file_path_split[-1] = str(counter)
            output_file_path = ".".join(output_file_path_split)

    dump_dada_file(output_file_path, header, data)


def create_parser():

    # current_dir = os.path.dirname(os.path.abspath(__file__))

    # config_dir = os.getenv("PFB_CONFIG_DIR",
    #                        os.path.join(current_dir, "config"))
    # data_dir = os.getenv("PFB_DATA_DIR",
    #                      os.path.join(current_dir, "data"))

    parser = argparse.ArgumentParser(
        description="add FIR filter info to existing DADA file")

    parser.add_argument("-i", "--input-file",
                        dest="input_file_path",
                        required=True)

    parser.add_argument("-f", "--fir-file",
                        dest="fir_file_path",
                        required=True)

    parser.add_argument("-c", "--channels",
                        dest="channels", default=8, type=int)

    parser.add_argument("-os", "--oversampling_factor",
                        dest="oversampling_factor", default="1/1", type=str)

    parser.add_argument("-ow", "--overwrite",
                        dest="overwrite", action="store_true")

    return parser


if __name__ == "__main__":
    parsed = create_parser().parse_args()
    # log_level = logging.INFO
    # if parsed.verbose:
    #     log_level = logging.DEBUG
    #
    # logging.basicConfig(level=log_level)
    # logging.getLogger("matplotlib").setLevel(logging.ERROR)

    _add_fir_data_to_existing_file(
        parsed.input_file_path,
        parsed.fir_file_path,
        parsed.oversampling_factor,
        parsed.channels,
        parsed.overwrite
    )
