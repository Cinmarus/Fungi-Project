from typing import Tuple
import numpy as np
import pandas as pd


def analyze_fft(data: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = len(data)

    fft_data = np.fft.fft(data)
    frequencies = np.fft.fftfreq(N, d=dt)

    # Normalize the amplitude for analysis (optional normalization)
    amplitudes = np.abs(fft_data) / N

    return frequencies, amplitudes, fft_data


def extract_baseline_and_signal(data: np.ndarray, frequencies: np.ndarray, fft_data: np.ndarray, f_cutoff: float) -> Tuple[np.ndarray, np.ndarray]:
    data = np.array(data, dtype=float)
    data = np.nan_to_num(data, nan=0.0)

    mask = np.abs(frequencies) <= f_cutoff
    filtered_fft = fft_data * mask

    baseline = np.fft.ifft(filtered_fft).real
    offset_signal = data - baseline

    print(baseline)

    return baseline, offset_signal


def extract_signal_from_data(df: pd.DataFrame, f_cutoff: float) -> pd.DataFrame:
    df = df.copy()

    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain a 'timestamp' column.")

    dt = np.median(np.diff(df["timestamp"].values))

    for col in df.columns:
        if col == "timestamp":
            continue

        data = df[col].values.astype(float)
        frequencies, amplitudes, fft_data = analyze_fft(data, dt)
        baseline, offset_signal = extract_baseline_and_signal(
            data, frequencies, fft_data, f_cutoff)

        df[f"{col}_baseline"] = baseline
        df[f"{col}_offset"] = offset_signal

    return df
