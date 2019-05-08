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

### v0.4.0

- Complete refactor. API is totally different. The code that computes the PFB
is its own function, with no dada wrangling wrapping.
- Added `format_handler.py` which defines a base class that is used to load
in input data, apply a transform, and then save to an output data file

### v0.5.0

- Added very basic, untested PFB inversion implementation, with suitable
subclass of `format_handler.FormatHandler` to load in real DADA data.
