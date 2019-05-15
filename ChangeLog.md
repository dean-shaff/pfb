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

### v0.5.1

- `format_handler.PSRFormatChannelizer` now copies `UTC_START` field from input
to channelized file
- `format_handler.PSRFormatSynthesizer` also copies `UTC_START` from input
to output file.

### v0.5.2

- `pfb_synthesis.pfb_synthesize` is now set up to be tested against real Matlab
test data.
- `pfb_synthesis.pfb_synthesize` works for streaming data in addition to single
chunks.
- `pfb_synthesis.pfb_synthesize` can use overlap discard, derippling, and
FFT windows. (not yet tested.)

### v0.6.0

- Added scripts for channelizing and synthesizing from the command line.
- Added some additional debug messages.
- Fixed issue where there wasn't 100% correspondance between matlab model
and Python PFB synthesis when applying derippling. The issue is sort of subtle,
but I can summarize with the following snippet:

```python
# before
>>> filt = np.arange(11)
>>> filt_wrong = np.append(filt[::-1], filt[:10])
>>> filt_correct = np.append(filt[1:][::-1], filt[:10])
>>> filt_wrong
array([10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0,  0,  1,  2,  3,  4,  5,
        6,  7,  8,  9, 10])
>>> filt_right
array([10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0,  1,  2,  3,  4,  5,  6,
        7,  8,  9])
```

The idea here is that there is no repeated sample at the border between the
two `append`ed chunks.
