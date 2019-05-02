### PFB Channelizer

Channelize input data using polyphase filterbank algorithm. Can use
critically sampled or oversampled data.

#### Installation

```
poetry install
```

#### Usage

From command line (not yet implemented)

```
python -m pfb.channelize -i <path-to-dada-file.dump> -c 256 -os 4/3
```

In a script:

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

output_data = pfb.pfb_analysis(input_data,
  os_factor="8/7", nchan=8, fir_filter_coeff=fir_filter_coeff)
```
