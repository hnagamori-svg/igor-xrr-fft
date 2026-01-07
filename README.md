# XRR FFT Thickness Analysis

X-ray Reflectivity (XRR) FFT analysis tool for determining thin film thickness.

## Based on

Lammel et al., Appl. Phys. Lett. 117, 213106 (2020)
"Fast Fourier transform and multi-Gaussian fitting of XRR data to determine the thickness of ALD grown thin films within the initial growth regime"
DOI: 10.1063/5.0024991

## Installation

1. Install Python dependencies:
```bash
pip install numpy scipy matplotlib
```

2. For Igor Pro integration:
   - Copy `igor_procedures/XRR_FFT_Analysis.ipf` to your Igor Procedures folder

## Usage

### Python GUI

```bash
cd python
python xrr_gui.py
```

### Python API

```python
from xrr_fft_core import analyze_xrr, XRRParameters
import numpy as np

# Load data
data = np.loadtxt('your_xrr_data.txt')
two_theta = data[:, 0]
intensity = data[:, 1]

# Set parameters
params = XRRParameters(
    wavelength=0.154,         # Cu Kalpha (nm)
    critical_angle=0.6,       # degrees
    interpolation_points=1000,
    zero_padding_factor=10
)

# Run analysis
result = analyze_xrr(two_theta, intensity, params)

# Get thickness results
if result.fit_result:
    print(f"Top layer: {result.fit_result.thickness_layer1:.2f} nm")
    print(f"Base layer: {result.fit_result.thickness_layer2:.2f} nm")
    print(f"Total: {result.fit_result.thickness_total:.2f} nm")
```

### Igor Pro

1. Load the procedure file
2. Go to Analysis -> XRR FFT Thickness Analysis
3. Select your 2\u03b8 and intensity waves
4. Click "Analyze"

## Algorithm Overview

1. **Q-space conversion**: Convert 2\u03b8 to Q with critical angle correction
2. **Fresnel correction**: Multiply by Q\^4 to compensate for Fresnel decay
3. **Interpolation**: Resample to equal Q spacing
4. **Windowing**: Apply Hamming (or other) window
5. **FFT**: Fast Fourier Transform with zero padding
6. **Multi-Gaussian fit**: Fit 3 Gaussians + 1/f\\\u03b1 noise

## Output

- Top layer thickness (nm)
- Base layer thickness (nm)
- Total thickness (nm)
- Fit quality (R^2)

## File Structure

```
/python
    xrr_fft_core.py      # Core analysis functions
    igor_interface.py    # Igor Pro interface
    xrr_gui.py           # GUI application
    generate_sample_data.py  # Sample data generator

/igor_procedures
    XRR_FFT_Analysis.ipf # Igor Pro procedure

/examples
    sample_xrr_data.txt  # Sample data file
```
