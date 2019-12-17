### PFB Channelizer

Channelize input data using polyphase filterbank algorithm. Can use
critically sampled or oversampled data.

#### Installation

```
poetry install
```

#### Usage


```python
import pfb

channelizer = pfb.PSRFormatChannelizer(
  os_factor="8/7", nchan=8, fir_filter_coeff="./path/to/filter_coeff")
output_file = channelizer("path/to/input_file")
```

Alternatively, if you have some numpy array you'd like to channelize,
simply use the `pfb_analysis` function:

```python

import pfb

output_data = pfb.pfb_analyze(input_data,
  os_factor="8/7", nchan=8, fir_filter_coeff=fir_filter_coeff)
```

As of version 0.5.0, there is now FFT based PFB inversion.


```python
import pfb

channelizer = pfb.PSRFormatChannelizer(
  os_factor="8/7", nchan=8, fir_filter_coeff="./path/to/filter_coeff")
output_file = channelizer("path/to/input_file")
```

Alternatively, if you have some numpy array you'd like to synthesize,
simply use the `pfb_synthesize` function:

```python

import pfb

output_data = pfb.pfb_synthesize(
  channelized_data,
  os_factor="8/7",
  deripple=True,
  input_overlap=32,
  input_fft_length=1024,
  fft_window=pfb.fft_windows.tukey_window(1024, 32)
)
```

A command line interface also exists for channelizing and synthesizing DADA
files.


```
python -m pfb.channelize -i <path-to-dada-file.dump> -c 256 -os 4/3
python -m pfb.channelize -i <path-to-dada-file.dump> --nchan 256 --os_factor 4/3
poetry run python -m pfb.channelize -i ~/ska/test_data/simulated_pulsar.noise_0.0.nseries_3.ndim_2.dump -c 8 -od ~/ska/test_data/ -f ~/mnt/ozstar/projects/PST_Matlab_dspsr_PFB_inversion_comparison/config/Prototype_FIR.4-3.8.80.mat -v
```

```
python -m pfb.synthesize -i <path-to-channelzied-dada-file.dump> -w tukey -x 1024 -p 32 -d
python -m pfb.synthesize -i <path-to-channelzied-dada-file.dump> --window tukey --fft_length 1024 --overlap 32 --deripple
```
