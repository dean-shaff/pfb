import os

import numpy as np

from pfb.fft_windows import tukey_window, top_hat_window

output_dir = "/home/SWIN/dshaff/ska/test_data"


def main():
    tukey_arr = tukey_window(1024, 128).astype(np.float32)
    top_hat_arr = top_hat_window(1024, 128).astype(np.float32)

    arr = [tukey_arr, top_hat_arr]
    file_names = ["tukey_window.dat", "tophat_window.dat"]

    for i in range(len(arr)):
        with open(os.path.join(output_dir, file_names[i]), 'wb') as f:
            arr[i].tofile(f)


if __name__ == "__main__":
    main()
