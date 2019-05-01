### v0.2.6

- Added PFB_DC_CHAN keyword to header.

### v0.3.0

- using `scipy.fftpack.ifft` instead of `numpy.fft.ifft` to compute Fourier
transforms. Numpy cannot compute a 64-bit FFT, while Scipy seems to be able to.
Note that Numpy will automatically upcast 32-bit data when applying
`numpy.fft.fft`.

### v0.3.1

- Separated channelizing from dumping data.
- Added psr_formats dependency

### v0.3.2

- `PFBChannelizer.from_input_files` now takes a `psr_formats.DataFile` instead
of a file path
- Added unittest for running `PFBChannelizer.from_input_files`