import numpy as np
from scipy.signal import butter, filtfilt, medfilt, savgol_filter
from scipy.fftpack import fft, ifft, fftfreq

from typing import Literal, Optional, Tuple, Union


BaselineMethods = Literal["fourier", "butterworth", "moving_average", "savgol"]
DenoiseMethods = Literal["lowpass", "median", "moving_average"]


def extract_baseline_and_offset(
        signal: np.ndarray,
        sampling_rate: float,
        method: BaselineMethods = "fourier",
        cutoff_freq: Optional[Union[float, Tuple[float, float]]] = 0.1,
        window_size: Optional[int] = 25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract baseline fluctuations and return the offset signal with one of
    three methods:
    1. Fourier transform with constrained frequencies
    2. Butterworth low-pass or band-pass filter
    3. Moving average
    4. Savitzky-Golay

    Returns the baseline fluctuations and the signal with the baseline subtracted.
    """
    if not (isinstance(cutoff_freq, float) or isinstance(cutoff_freq, tuple)):
        raise TypeError("cutoff_freq must be a float or a tuple")

    if method == "fourier":
        n = len(signal)
        freqs = fftfreq(n, d=1 / sampling_rate)
        fft_signal = fft(signal)

        if isinstance(cutoff_freq, tuple):
            low, high = cutoff_freq
            fft_signal[(np.abs(freqs) < low) | (np.abs(freqs) > high)] = 0
        else:
            fft_signal[np.abs(freqs) > cutoff_freq] = 0

        offset_signal = np.real(ifft(fft_signal))
        baseline = signal - offset_signal

        return baseline, offset_signal
    elif method == "butterworth":
        nyquist = 0.5 * sampling_rate

        if isinstance(cutoff_freq, tuple):
            low, high = cutoff_freq
            normal_cutoff = [low / nyquist, high / nyquist]
            b, a = butter(4, normal_cutoff, btype="bandpass", analog=False)
        else:
            normal_cutoff = cutoff_freq / nyquist
            b, a = butter(4, normal_cutoff, btype="lowpass", analog=False)

        baseline = filtfilt(b, a, signal)
    elif method == "moving_average":
        if window_size is None:
            raise ValueError(
                f"For method '{method}', 'window_size' must be specified")

        baseline = np.convolve(signal, np.ones(
            window_size) / window_size, mode="same")
    elif method == "savgol":
        if window_size is None:
            raise ValueError(f"The Savitzky-Golay filter requires a window size to be specified!")
        
        baseline = savgol_filter(signal, window_size, 2)
    else:
        raise ValueError(
            f"Invalid method '{method}'. Choose from: 'fourier', 'butterworth', or 'moving_average'.")

    offset_signal = signal - baseline

    return baseline, offset_signal


def denoise_signal(
        signal: np.ndarray,
        sampling_rate: float,
        method: DenoiseMethods,
        cutoff_freq: Optional[float],
        window_size: Optional[int],
) -> np.ndarray:
    if method == "lowpass":
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(4, normal_cutoff, btype="lowpass", analog=False)
        denoised_signal = filtfilt(b, a, signal)
    elif method == "median":
        denoised_signal = medfilt(signal, kernel_size=window_size)
    elif method == "moving_average":
        denoised_signal = np.convolve(signal, np.ones(
            window_size) / window_size, mode="same")
    else:
        raise ValueError(
            f"Invalid method '{method}'. Choose from: 'lowpass', 'median', or 'moving_average'.")

    return denoised_signal
