import logging
import typing

import numpy as np
import scipy.io

from .rational import Rational

module_logger = logging.getLogger(__name__)


__all__ = [
    "dtype_to_int",
    "complex_dtype_lookup",
    "float_dtype_lookup",
    "fir_filter_coeff_type",
    "os_factor_type",
    "filter_info_to_dict",
    "load_matlab_filter_coeff",
    "filter_coeff_to_str",
    "str_to_filter_coeff"
]


dtype_to_int = {
    np.dtype(np.float32): 1,
    np.dtype(np.float64): 1,
    np.dtype(np.complex64): 2,
    np.dtype(np.complex128): 2
}

complex_dtype_lookup = {
    np.dtype(np.float32): np.dtype(np.complex64),
    np.dtype(np.float64): np.dtype(np.complex128),
    np.dtype(np.complex64): np.dtype(np.complex64),
    np.dtype(np.complex128): np.dtype(np.complex128)
}

float_dtype_lookup = {
    np.dtype(np.float32): np.dtype(np.float32),
    np.dtype(np.float64): np.dtype(np.float64),
    np.dtype(np.complex64): np.dtype(np.float32),
    np.dtype(np.complex128): np.dtype(np.float64)
}

fir_filter_coeff_type = typing.Union[str, np.ndarray]
os_factor_type = typing.Union[str, Rational]
overlap_type = typing.Union[int, typing.Callable]


def filter_info_to_dict(
    filter_info: typing.List[dict]
) -> dict:
    header = {}
    nstage = len(filter_info)
    header["NSTAGE"] = nstage
    for i in range(nstage):
        filter_coeff = filter_info[i]["COEFF"]
        filter_coeff_str = ",".join(
            filter_coeff_to_str(filter_coeff))

        header[f"OVERSAMP_{i}"] = filter_info[i]["OVERSAMP"]
        header[f"NTAP_{i}"] = len(filter_coeff)
        header[f"COEFF_{i}"] = filter_coeff_str
        header[f"NCHAN_PFB_{i}"] = filter_info[i]["NCHAN_PFB"]
    return header


def load_matlab_filter_coeff(file_path: str) -> typing.Tuple:
    fir_config = scipy.io.loadmat(file_path)
    fir_filter_coeff = fir_config["h"].reshape(-1)
    return fir_config, fir_filter_coeff


def filter_coeff_to_str(filter_coeff: np.ndarray) -> typing.List[str]:
    """
    Given some filter coefficients, dump them to ascii format.

    Returns:
        list: a list of strings
    """
    module_logger.debug(f"filter_coeff_to_str: filter_coeff={filter_coeff}")
    filter_coeff_as_ascii = ["{:.6E}".format(n) for n in filter_coeff]
    return filter_coeff_as_ascii


def str_to_filter_coeff(filter_coeff_str: str, delimiter: str = ",") -> np.ndarray:
    """
    Given some filter coefficients, represented as a string of ascii numbers,
    create a numpy array.
    """
    module_logger.debug((f"str_to_filter_coeff: "
                         f"filter_coeff_str={filter_coeff_str}, "
                         f"delimiter={delimiter}"))
    filter_coeff = [float(s) for s in filter_coeff_str.split(delimiter)]
    module_logger.debug(f"str_to_filter_coeff: filter_coeff={filter_coeff}")
    return np.asarray(filter_coeff)


# def add_fir_data_to_existing_file(
#     file_path: str,
#     fir_file_path: str,
#     os_factor: str,
#     channels: int,
#     overwrite: bool = False
# ) -> None:
#     _, coeff = load_matlab_filter_coeff(fir_file_path)
#
#     fir_info = [{
#         "COEFF": coeff,
#         "NTAPS": len(coeff),
#         "OVERSAMP": str(os_factor),
#         "NCHAN_PFB": channels
#     }]
#
#     header, data = load_dada_file(file_path)
#     header.update(filter_info_to_dict(fir_info))
#     output_file_path = file_path
#     counter = 0
#     if not overwrite:
#         output_file_path = f"{output_file_path}.{counter}"
#         while os.path.exists(output_file_path):
#             counter += 1
#             output_file_path_split = output_file_path.split(".")
#             output_file_path_split[-1] = str(counter)
#             output_file_path = ".".join(output_file_path_split)
#
#     dump_dada_file(output_file_path, header, data)
#
#
# def create_parser():
#
#     # current_dir = os.path.dirname(os.path.abspath(__file__))
#
#     # config_dir = os.getenv("PFB_CONFIG_DIR",
#     #                        os.path.join(current_dir, "config"))
#     # data_dir = os.getenv("PFB_DATA_DIR",
#     #                      os.path.join(current_dir, "data"))
#
#     parser = argparse.ArgumentParser(
#         description="add FIR filter info to existing DADA file")
#
#     parser.add_argument("-i", "--input-file",
#                         dest="input_file_path",
#                         required=True)
#
#     parser.add_argument("-f", "--fir-file",
#                         dest="fir_file_path",
#                         required=True)
#
#     parser.add_argument("-c", "--channels",
#                         dest="channels", default=8, type=int)
#
#     parser.add_argument("-os", "--oversampling_factor",
#                         dest="oversampling_factor", default="1/1", type=str)
#
#     parser.add_argument("-ow", "--overwrite",
#                         dest="overwrite", action="store_true")
#
#     return parser
#
#
# if __name__ == "__main__":
#     parsed = create_parser().parse_args()
#     # log_level = logging.INFO
#     # if parsed.verbose:
#     #     log_level = logging.DEBUG
#     #
#     # logging.basicConfig(level=log_level)
#     # logging.getLogger("matplotlib").setLevel(logging.ERROR)
#
#     add_fir_data_to_existing_file(
#         parsed.input_file_path,
#         parsed.fir_file_path,
#         parsed.oversampling_factor,
#         parsed.channels,
#         parsed.overwrite
#     )
