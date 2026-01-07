"""
XRR FFT Thickness Analysis Package
"""

from .xrr_fft_core import (
    XRRParameters,
    WindowType,
    FFTResult,
    GaussianFitResult,
    XRRAnalysisResult,
    analyze_xrr,
    detect_critical_angle,
    XRR,
    FFT,
)

from .igor_interface import IgorProInterface, XRRIgorBridge, IgorWave

__all__ = [
    'XRRParameters',
    'WindowType',
    'FFTResult',
    'GaussianFitResult',
    'XRRAnalysisResult',
    'analyze_xrr',
    'detect_critical_angle',
    'XRR',
    'FFT',
    'IgorProInterface',
    'XRRIgorBridge',
    'IgorWave',
]
