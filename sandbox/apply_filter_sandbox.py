import numpy as np
import scipy.signal
import scipy.fftpack
import matplotlib.pyplot as plt


def pad_filter(filter_coeff, downsample_by):
    init_filter_size = filter_coeff.shape[0]
    rem = init_filter_size % downsample_by
    if rem != 0:
        filter_coeff_padded = np.zeros(
            (init_filter_size + downsample_by - rem),
            dtype=filter_coeff.dtype
        )
        # print(filter_coeff_padded.shape)
        # filter_coeff_padded[init_filter_size:] = filter_coeff
        filter_coeff_padded[:init_filter_size] = filter_coeff
        filter_coeff = filter_coeff_padded

    return filter_coeff


def apply_filter(signal: np.ndarray,
                 filter_coeff: np.ndarray,
                 filtered: np.ndarray,
                 downsample_by: int,
                 increment_by: int = None) -> np.ndarray:
    """
    filter signal with filter_coeff.

    Assume complex64 `signal`, float32 `filter_coeff`
    """
    if increment_by is None:
        increment_by = downsample_by

    signal_dtype = signal.dtype

    filter_coeff = pad_filter(filter_coeff, downsample_by)
    yield filter_coeff
    window_size = filter_coeff.shape[0]

    down_sample_filter_elem = int(window_size / downsample_by)
    filter_idx = np.arange(window_size).reshape(
        (down_sample_filter_elem, downsample_by))

    filter_coeff_2d = filter_coeff.reshape((-1, downsample_by))
    # filter_coeff_2d = np.zeros(filter_idx.shape, dtype=filter_coeff.dtype)
    # for i in range(downsample_by):
    #     filter_coeff_2d[:, i] = filter_coeff[filter_idx[:, i]]

    # yield filter_coeff_2d
    # yield filter_coeff.reshape((-1, downsample_by))

    signal_padded = np.zeros(
        (window_size + signal.shape[0]),
        dtype=signal_dtype
    )
    signal_padded[window_size:] = signal

    down_sample_signal_elem = filtered.shape[0]
    signal_chunk_2d = np.zeros(filter_idx.shape, dtype=signal_dtype)

    for i in range(down_sample_signal_elem):
        idx = i*increment_by
        signal_chunk = signal_padded[idx:idx + window_size][::-1]
        # filtered[i,:] = np.sum(
        #   signal_chunk[filter_idx] * filter_coeff_2d, axis=0)
        for j in range(downsample_by):
            signal_chunk_2d[:, j] = signal_chunk[filter_idx[:, j]]

        filtered[i, :] = (
            np.sum(signal_chunk_2d.real * filter_coeff_2d, axis=0) +
            1j*np.sum(signal_chunk_2d.imag * filter_coeff_2d, axis=0)
        )

    yield filtered


def convolve(a, b, n):

    return scipy.fftpack.ifft(
        scipy.fftpack.fft(a, n) *
        scipy.fftpack.fft(b, n)
    )


def cross_correlation(a, b):
    return scipy.signal.fftconvolve(a, b[::-1])


def apply_filter_fft_alt(signal: np.ndarray,
                         filter_coeff: np.ndarray,
                         filtered: np.ndarray,
                         downsample_by: int,
                         increment_by: int = None):
    if increment_by is None:
        increment_by = downsample_by

    signal_dtype = signal.dtype
    filter_dtype = filter_coeff.dtype

    filter_coeff = pad_filter(filter_coeff, downsample_by)
    window_size = filter_coeff.shape[0]
    down_sample_signal_elem = int(signal.shape[0] / downsample_by)
    signal_padded = np.zeros(
        (window_size + signal.shape[0]),
        dtype=signal_dtype
    )
    signal_padded = np.append(
        np.zeros(
            window_size,
            dtype=signal_dtype
        ),
        signal[:down_sample_signal_elem*downsample_by]
        # np.conj(
        # signal[:down_sample_signal_elem*downsample_by]
        # )
    )
    down_sample_signal_padded_elem = int(signal_padded.shape[0] / downsample_by)

    delta = down_sample_signal_padded_elem - down_sample_signal_elem

    filtered = convolve(signal_padded, filter_coeff,
                        down_sample_signal_padded_elem*downsample_by)

    filtered = filtered.reshape((
        down_sample_signal_padded_elem, downsample_by))
    filtered = filtered[delta:down_sample_signal_elem+delta, :]
    filtered /= downsample_by
    yield filtered


def apply_filter_fft(signal: np.ndarray,
                     filter_coeff: np.ndarray,
                     filtered: np.ndarray,
                     downsample_by: int,
                     increment_by: int = None):

    if increment_by is None:
        increment_by = downsample_by

    signal_dtype = signal.dtype
    filter_dtype = filter_coeff.dtype

    # filter_coeff = pad_filter(filter_coeff, downsample_by)
    window_size = filter_coeff.shape[0]
    down_sample_signal_elem = int(signal.shape[0] / downsample_by)
    signal_padded = np.zeros(
        (window_size + signal.shape[0]),
        dtype=signal_dtype
    )
    signal_padded = np.append(
        np.zeros(
            window_size,
            dtype=signal_dtype
        ),
        # signal
        signal[:down_sample_signal_elem*downsample_by]
        # np.conj(
        # signal[:down_sample_signal_elem*downsample_by]
        # )
    )
    down_sample_signal_padded_elem = int(signal_padded.shape[0] / downsample_by)
    # signal_padded[:-window_size] = np.conj(signal)[::-1]
    # signal_padded[:-window_size] = signal
    # signal_padded[window_size:] = signal
    # signal_padded = signal

    # down_sample_signal_elem = int(signal_padded.shape[0] / downsample_by)
    # down_sample_filter_elem = int(filter_coeff.shape[0] / downsample_by)

    # print(f"down_sample_signal_elem={down_sample_signal_elem}")
    # print(f"down_sample_filter_elem={down_sample_filter_elem}")

    # filter_downsampled = np.zeros(down_sample_signal_elem,
    #                               dtype=filter_dtype)
    #
    # signal_downsampled = np.zeros(down_sample_signal_elem,
    #                               dtype=signal_dtype)
    print(down_sample_signal_padded_elem)
    for ichan in range(downsample_by):
        # filter_downsampled[:down_sample_filter_elem] = \
        #     filter_coeff[ichan::downsample_by]
        # signal_downsampled[:down_sample_signal_elem] = \
        #     signal_padded[ichan::downsample_by]
        # print(f"filter_downsampled.shape={filter_downsampled.shape}")
        # print(f"signal_downsampled.shape={signal_downsampled.shape}")
        # print(f"filtered.shape={filtered.shape}")
        # convolved = scipy.fftpack.ifft(
        #     scipy.fftpack.fft(signal_padded[ichan::downsample_by],
        #         down_sample_signal_padded_elem) *
        #     scipy.fftpack.fft(filter_coeff[ichan::downsample_by],
        #         down_sample_signal_padded_elem)
        # )
        # convolved = convolve(
        #     signal_padded[ichan::downsample_by],
        #     filter_coeff[ichan::downsample_by],
        #     down_sample_signal_padded_elem
        # )
        convolved = scipy.signal.fftconvolve(
            signal_padded[ichan::downsample_by],
            filter_coeff[ichan::downsample_by], "full")
        # print(convolved_scipy.shape)
        # print(filtered.shape[0])
        delta = convolved.shape[0] - filtered.shape[0]
        delta_2 = int(delta / 2)
        # print(np.allclose(convolved_np, convolved_scipy))

        # if ichan % 2 != 0:
        #     convolved_scipy = -convolved_scipy
        # filtered[:, ichan] = convolved_scipy[delta_2:down_sample_signal_elem+delta_2]
        filtered[:, ichan] = convolved[delta_2:down_sample_signal_elem+delta_2]
        # filtered[:, ichan] = convolved[delta:down_sample_signal_elem+delta]

    yield filtered


def main():

    nchan = 8
    filter_coeff = np.arange(155).astype(np.float32)
    t = np.arange(500)
    signal = np.sin(t/(5*np.pi)).astype(np.float32)
    # signal = np.random.randn(1000).astype(np.float32)
    signal = signal + 1j*signal

    def filtered():
        return np.zeros((int(signal.shape[0] / nchan), nchan),
                        dtype=np.complex64)

    filterer = apply_filter_fft_alt(
        signal.copy(), filter_coeff, filtered(), nchan, None)

    for i in filterer:
        filtered_fft = i

    filterer = apply_filter(
        signal.copy(), filter_coeff, filtered(), nchan, None)

    for i in filterer:
        filtered_no_fft = i

    plt.ion()
    fig, axes = plt.subplots(4, 1)
    # print(np.allclose(filtered_no_fft, filtered_fft, atol=1e-2, rtol=1e-2))
    for ichan in range(nchan):
        fft_ichan = filtered_fft[:, ichan]
        no_fft_ichan = filtered_no_fft[:, ichan]

        print(np.allclose(fft_ichan, no_fft_ichan, atol=1e-3))
        for ax in axes:
            ax.grid(True)

        mid = fft_ichan.shape[0]
        axes[0].plot(np.abs(no_fft_ichan), color="r")
        axes[1].plot(np.angle(no_fft_ichan), color="r")
        axes[0].plot(np.abs(fft_ichan), color="g")
        axes[1].plot(np.angle(fft_ichan), color="g")
        xcorr = cross_correlation(fft_ichan, no_fft_ichan)
        offset = np.abs(np.argmax(np.abs(xcorr)) - mid)
        print(f"ichan={ichan}, offset={offset}")
        axes[2].plot(np.abs(xcorr))
        diff = fft_ichan - no_fft_ichan
        axes[3].plot(np.abs(diff))
        axes[3].plot(np.angle(diff))
        input(">> ")
        # plt.show()


if __name__ == "__main__":
    main()
