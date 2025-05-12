import numpy as np
from scipy.signal import savgol_filter


MOVING_AVERAGE_WINDOW_SIZE = 5000
SAVITZKY_GOLAY_WINDOW_SIZE = 50
SAVITZKY_GOLAY_POLY_ORDER = 3


def apply_filtering(time: np.ndarray, signal: np.ndarray) -> np.ndarray:
    sampling_rate = 1 / np.mean(np.diff(time))

    # first, extract the baseline by using a moving average with a large window size
    moving_average_baseline = np.convolve(signal, np.ones(
        MOVING_AVERAGE_WINDOW_SIZE) / MOVING_AVERAGE_WINDOW_SIZE, mode="same")

    # next, get the offset signal by subtracting the baseline from the raw signal
    offset_signal = signal - moving_average_baseline

    # finally, apply the Savitzky-Golay filter to denoise
    denoised_offset = savgol_filter(
        offset_signal, SAVITZKY_GOLAY_WINDOW_SIZE, SAVITZKY_GOLAY_POLY_ORDER)

    return denoised_offset
