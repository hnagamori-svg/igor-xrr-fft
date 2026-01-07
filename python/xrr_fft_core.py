"""
XRR FFT Thickness Analysis Core Module

Paper: Lammel et al., Appl. Phys. Lett. 117, 213106 (2020)
DOI: 10.1063/5.0024991
ArXdF: 2008.04626v2
"""

import numpy as np
from scipy import interpolate, signal, fftpack
from scipy.optimize import curve_fit
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class WindowType(Enum):
    NONE = 0
    HANNING = 1
    HAMMING = 2
    FLATTOP = 3


@dataclass
class XRRParameters:
    wavelength: float = 0.154
    critical_angle: float = 0.6
    interpolation_points: int = 1000
    zero_padding_factor: int = 10
    window_type: WindowType = WindowType.HAMMING


@dataclass
class FFTResult:
    freq: np.ndarray
    amplitude: np.ndarray
    amplitude_raw: np.ndarray


@dataclass
class GaussianFitResult:
    thickness_layer1: float
    thickness_layer2: float
    thickness_total: float
    params: np.ndarray
    params_cov: Optional[np.ndarray]
    x_fit: np.ndarray
    y_fit: np.ndarray
    y_gauss1: np.ndarray
    y_gauss2: np.ndarray
    y_gauss3: np.ndarray
    y_noise: np.ndarray
    r_squared: float
    residual_std: float


def xrr_to_q_space(two_theta, intensity, critical_angle, wavelength=0.152):
    s_cor = 2 * np.sqrt(
        (np.cos(np.pi * critical_angle / 2 / 180))**2 - 
        (np.cos(np.pi * two_theta / 2 / 180))**2
    ) / wavelength
    intensity_corrected = s_cor**4 * intensity
    return s_cor, intensity_corrected


def interpolate_to_equal_spacing(s_cor, intensity, n_points=1000):
    mask = np.logical_not(np.isnan(s_cor))
    s_valid = s_cor[mask]
    intensity_valid = intensity[mask]
    if len(s_valid) < 4:
        raise ValueError("Not enough data points above critical angle")
    x = np.linspace(s_valid.min(), s_valid.max(), n_points)
    f = interpolate.interp1d(s_valid, intensity_valid, kind="cubic")
    return x, f(x)


def apply_window(y, window_type):
    N = len(y)
    if window_type == WindowType.NONE:
        window = np.ones(N)
    elif window_type == WindowType.HANNING:
        window = signal.windows.hann(N)
    elif window_type == WindowType.HAMMING:
        window = signal.windows.hamming(N)
    elif window_type == WindowType.FLATTOP:
        window = signal.windows.flattop(N)
    else:
        window = np.ones(N)
    return window * y / np.mean(window)


def perform_fft(x, y, window_type=WindowType.HAMMING, zero_padding_factor=10):
    N = len(y)
    d = x[1] - x[0]
    n_pad = N * zero_padding_factor
    y_windowed = apply_window(y, window_type)
    yf = (2/N) * np.abs(fftpack.fft(y_windowed, n=n_pad))
    xf = fftpack.fftfreq(n_pad, d=d)
    xf = xf[:n_pad//2]
    yf = yf[:n_pad//2]
    yf_normalized = yf / yf[0] if yf[0] != 0 else yf
    return FFTResult(freq=xf, amplitude=yf_normalized, amplitude_raw=yf)


def XRR(data, crit_ang, wavelength=0.152, n_points=1000):
    two_theta = data[:, 0]
    intensity = data[:, 1]
    s_cor, intensity_corrected = xrr_to_q_space(two_theta, intensity, crit_ang, wavelength)
    return interpolate_to_equal_spacing(s_cor, intensity_corrected, n_points)


def FFT(x, y, d=None, window=2, n=None):
    if d is None:
        d = x[1] - x[0]
    N = len(y)
    if window == 0:
        win = np.ones(N)
    elif window == 1:
        win = signal.windows.hann(N)
    elif window == 2:
        win = signal.windows.hamming(N)
    else:
        win = signal.windows.flattop(N)
    if n is None:
        n = N
    yf = (2/N) * np.abs(fftpack.fft(win * y / np.mean(win), n=n))
    xf = fftpack.fftfreq(n, d=d)
    return xf[:n//2], yf[:n//2]


def func_noise(x, amp, ex):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = amp / np.power(x, ex)
        result = np.where(np.isfinite(result), result, 0)
    return result


def func_gauss(p, a, pmax, w):
    return a * np.exp(-np.log(2) * ((pmax - p) / (w / 2))**2)


def func_gauss3(p, a1, w1, a2, pmax2, w2, a3, pmax3, w3, amp, ex, z0):
    pmax1 = pmax3 - pmax2
    return (func_gauss(p, a1, pmax1, w1) + func_gauss(p, a2, pmax2, w2) +
            func_gauss(p, a3, pmax3, w3) + func_noise(p, amp, ex) + z0)


def fit_noise_background(xf, yf, mask_ranges=None):
    if mask_ranges is None:
        mask_ranges = [(1, 5), (26, 80)]
    mask = np.zeros(len(xf), dtype=bool)
    for rmin, rmax in mask_ranges:
        mask |= (xf > rmin) & (xf < rmax)
    if np.sum(mask) < 3:
        return 1.0, 2.0
    try:
        popt, _ = curve_fit(func_noise, xf[mask], yf[mask], 
                            p0=[1.0, 2.0], bounds=(0, np.inf), maxfev=5000)
        return popt[0], popt[1]
    except:
        return 1.0, 2.0


def fit_multi_gaussian(xf, yf, initial_guess=None, fit_range=(1.1, 80),
                       noise_mask_ranges=None):
    if noise_mask_ranges is None:
        noise_mask_ranges = [(1, 5), (26, 80)]
    mask = (xf > fit_range[0]) & (xf < fit_range[1])
    xf_fit, yf_fit = xf[mask], yf[mask]
    if len(xf_fit) < 12:
        raise ValueError("Not enough data points in fit range")
    
    amp_init, ex_init = fit_noise_background(xf, yf, noise_mask_ranges)
    if initial_guess is None:
        initial_guess = [0.2, 0.3, 0.2, 7.0, 0.3, 0.2, 13.0, 0.3, amp_init, ex_init, 2e-3]
    
    popt, pcov = curve_fit(func_gauss3, xf_fit, yf_fit, p0=initial_guess,
                           bounds=(0, np.inf), maxfev=10000)
    
    a1, w1, a2, pmax2, w2, a3, pmax3, w3, amp, ex, z0 = popt
    pmax1 = pmax3 - pmax2
    
    x_dense = np.linspace(xf_fit.min(), xf_fit.max(), 500)
    y_pred = func_gauss3(xf_fit, *popt)
    ss_res = np.sum((yf_fit - y_pred)**2)
    ss_tot = np.sum((yf_fit - np.mean(yf_fit))**2)
    
    return GaussianFitResult(
        thickness_layer1=pmax1, thickness_layer2=pmax2, thickness_total=pmax3,
        params=popt, params_cov=pcov, x_fit=x_dense,
        y_fit=func_gauss3(x_dense, *popt),
        y_gauss1=func_gauss(x_dense, a1, pmax1, w1),
        y_gauss2=func_gauss(x_dense, a2, pmax2, w2),
        y_gauss3=func_gauss(x_dense, a3, pmax3, w3),
        y_noise=func_noise(x_dense, amp, ex) + z0,
        r_squared=1 - (ss_res / ss_tot),
        residual_std=np.std(yf_fit - y_pred)
    )


@dataclass
class XRRAnalysisResult:
    two_theta_raw: np.ndarray
    intensity_raw: np.ndarray
    q_space: np.ndarray
    intensity_corrected: np.ndarray
    q_interp: np.ndarray
    intensity_interp: np.ndarray
    fft_result: FFTResult
    fit_result: Optional[GaussianFitResult]
    parameters: XRRParameters


def analyze_xrr(two_theta, intensity, params=None, do_fitting=True, fit_initial_guess=None):
    if params is None:
        params = XRRParameters()
    
    q_space, intensity_corrected = xrr_to_q_space(
        two_theta, intensity, params.critical_angle, params.wavelength)
    q_interp, intensity_interp = interpolate_to_equal_spacing(
        q_space, intensity_corrected, params.interpolation_points)
    fft_result = perform_fft(q_interp, intensity_interp,
                             params.window_type, params.zero_padding_factor)
    
    fit_result = None
    if do_fitting:
        try:
            fit_result = fit_multi_gaussian(fft_result.freq, fft_result.amplitude,
                                           initial_guess=fit_initial_guess)
        except Exception as e:
            print(f"Warning: Fitting failed: {e}")
    
    return XRRAnalysisResult(
        two_theta_raw=two_theta, intensity_raw=intensity,
        q_space=q_space, intensity_corrected=intensity_corrected,
        q_interp=q_interp, intensity_interp=intensity_interp,
        fft_result=fft_result, fit_result=fit_result, parameters=params
    )


def detect_critical_angle(two_theta, intensity):
    max_intensity = np.max(intensity)
    half_max = max_intensity / 2
    max_idx = np.argmax(intensity)
    for i in range(max_idx, len(intensity)):
        if intensity[i] < half_max:
            if i > 0:
                frac = (half_max - intensity[i-1]) / (intensity[i] - intensity[i-1])
                return two_theta[i-1] + frac * (two_theta[i] - two_theta[i-1])
            return two_theta[i]
    return 0.6
