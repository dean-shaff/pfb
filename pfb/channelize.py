import argparse
import os
import logging

import psr_formats

from .format_handler import PSRFormatChannelizer


def create_parser():

    parser = argparse.ArgumentParser(
        description="channelize data")

    parser.add_argument("-i", "--input-file",
                        dest="input_file_path",
                        required=True)

    parser.add_argument("-f", "--fir-file",
                        dest="fir_file_path",
                        required=True)

    parser.add_argument("-v", "--verbose",
                        dest="verbose", action="store_true")

    parser.add_argument("-c", "--nchan",
                        dest="nchan", default=8, type=int)

    parser.add_argument("-os", "--os_factor",
                        dest="os_factor", default="4/3", type=str)

    parser.add_argument("-o", "--output_file_name",
                        dest="output_file_name", default="", type=str)

    parser.add_argument("-od", "--output_dir",
                        dest="output_dir", default="./", type=str)

    parser.add_argument("--use-ifft",
                        dest="use_ifft", action="store_true")

    parser.add_argument("--pfb-dc-chan",
                        dest="pfb_dc_chan", action="store_true")

    return parser


if __name__ == "__main__":

    parsed = create_parser().parse_args()
    level = logging.ERROR
    if parsed.verbose:
        level = logging.DEBUG
    logging.basicConfig(level=level)

    channelizer = PSRFormatChannelizer(
        os_factor=parsed.os_factor,
        nchan=parsed.nchan,
        fir_filter_coeff=parsed.fir_file_path,
        use_ifft=parsed.use_ifft
    )

    output_file_name = parsed.output_file_name
    if output_file_name == "":
        output_file_name = ("channelized." +
                            os.path.basename(parsed.input_file_path))
    dada_file = psr_formats.DADAFile(parsed.input_file_path)

    if parsed.pfb_dc_chan:
        dada_file["PFB_DC_CHAN"] = "1"

    channelizer(dada_file,
                output_dir=parsed.output_dir,
                output_file_name=output_file_name)
