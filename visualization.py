from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis import BaselineMethods, DenoiseMethods, denoise_signal, extract_baseline_and_offset
from Data_visualisation import set_plt_defaults


def visualize_data(df: pd.DataFrame) -> None:
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain a 'timestamp' column")

    time = df["timestamp"]

    for col in df.columns:
        if col != "timestamp":
            plt.plot(time, df[col], label=col)

    plt.legend()

    plt.show()


def visualize_data_baseline(df: pd.DataFrame, column: str, timestamp_column: str = "timestamp") -> None:
    fig, ax = plt.subplots()

    raw_line, = ax.plot(
        df[timestamp_column],
        df[column],
        label="Raw Data",
        linewidth=2,
    )

    baseline_color = raw_line.get_color()
    baseline_col = f"{column}_baseline"

    ax.plot(
        df[timestamp_column],
        df[baseline_col],
        label="Baseline",
        linestyle="--",
        linewidth=2,
    )

    ax.legend()
    ax.grid(True)

    plt.show()


def visualize_data_offset(df: pd.DataFrame, column: str, timestamp_column: str = "timestamp") -> None:
    fig, (ax1, ax2) = plt.subplots(2, sharey=True, sharex=True)

    baseline_col = f"{column}_offset"

    ax1.plot(
        df[timestamp_column],
        df[baseline_col],
    )

    ax1.grid(True)
    ax1.set_title("Offset from baseline")

    ax2.plot(
        df[timestamp_column],
        df[column],
    )

    ax2.grid(True)
    ax2.set_title("Raw data")

    plt.show()


def create_baseline_plot(
        time: np.ndarray,
        signal: np.ndarray,
        sampling_rate: float,
        method: BaselineMethods = "fourier",
        cutoff_freq: Optional[Union[float, Tuple[float, float]]] = 0.1,
        window_size: Optional[int] = 50,
) -> None:
    baseline, offset = extract_baseline_and_offset(
        signal,
        sampling_rate,
        method=method,
        cutoff_freq=cutoff_freq,
        window_size=window_size
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(time, signal, label="Raw Signal")
    ax.plot(time, baseline, label="Baseline Signal")

    if method == "fourier" or method == "butterworth":
        if isinstance(cutoff_freq, tuple):
            extra_info = f"low frequency = {cutoff_freq[0]} Hz, high frequency = {cutoff_freq[1]} Hz"
            file_meta = f"bandpass={cutoff_freq[0]}-{cutoff_freq[1]}"
        else:
            extra_info = f"high frequency = {cutoff_freq} Hz"
            file_meta = f"lowpass={cutoff_freq}"
    else:
        extra_info = f"window size = {window_size} samples"
        file_meta = f"size={window_size}"

    current_title = f"Baseline Extraction (method = {method}, {extra_info})"

    ax.legend()
    ax.grid()
    ax.set_title(current_title)

    plt.savefig(
        f"output/baseline_plots/baseline_{method}_{file_meta}.png", dpi=300)


def create_offset_plot(
        time: np.ndarray,
        signal: np.ndarray,
        sampling_rate: float,
        baseline_method: BaselineMethods = "fourier",
        cutoff_freq: Optional[Union[float, Tuple[float, float]]] = 0.1,
        window_size: Optional[int] = 50,
) -> None:
    baseline, offset = extract_baseline_and_offset(
        signal,
        sampling_rate,
        method=baseline_method,
        cutoff_freq=cutoff_freq,
        window_size=window_size
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(time, signal, label="Raw Signal")
    ax.plot(time, offset, label="Offset Signal")

    if baseline_method == "fourier" or baseline_method == "butterworth":
        if isinstance(cutoff_freq, tuple):
            extra_info = f"low frequency = {cutoff_freq[0]} Hz, high frequency = {cutoff_freq[1]} Hz"
            file_meta = f"bandpass={cutoff_freq[0]}-{cutoff_freq[1]}"
        else:
            extra_info = f"high frequency = {cutoff_freq} Hz"
            file_meta = f"lowpass={cutoff_freq}"
    else:
        extra_info = f"window size = {window_size} samples"
        file_meta = f"size={window_size}"

    current_title = f"Signal Offset Extraction (method = {baseline_method}, {extra_info})"

    ax.legend()
    ax.grid()
    ax.set_title(current_title)

    plt.savefig(
        f"output/offset_plots/offset_{baseline_method}_{file_meta}.png", dpi=300)


def create_denoised_plot(
        time: np.ndarray,
        signal: np.ndarray,
        sampling_rate: float,
        baseline_method: BaselineMethods = "fourier",
        denoise_method: DenoiseMethods = "Lowpass",
        cutoff_freq: Union[float, Tuple[float, float]] = 0.1,
        noise_cutoff: float = 10,
        window_size: int = 50,
) -> None:
    baseline, offset = extract_baseline_and_offset(
        signal,
        sampling_rate,
        method=baseline_method,
        cutoff_freq=cutoff_freq,
        window_size=window_size
    )

    denoised_offset = denoise_signal(
        offset,
        sampling_rate,
        method=denoise_method,
        cutoff_freq=noise_cutoff,
        window_size=window_size
    )

    df_offset = pd.DataFrame({'timestamp': time, 'signal_without_baseline': offset})
    df_offset.to_csv("data/signal_without_baseline.csv", index=False)
    print("offset signal saved to data/signal_without_baseline.csv")

    fig, ax = plt.subplots(figsize=(12, 8))

    # ax.plot(time, signal, label="Raw Signal")
    ax.plot(time, offset, label="Offset Signal")
    ax.plot(time, denoised_offset, label="Denoised Offset Signal")

    if denoise_method == "lowpass" or denoise_method == "median":
        extra_info = f"cutoff frequency = {cutoff_freq} Hz"
        file_meta = f"cutoff={cutoff_freq}"
    else:
        extra_info = f"window size = {window_size} samples"
        file_meta = f"size={window_size}"

    current_title = f"Signal Denoising (method = {denoise_method}, {extra_info})"

    ax.legend()
    ax.grid()
    ax.set_title(current_title)

    plt.show()
    print("test")
    print(f"output/denoise_plots/denoise_{denoise_method}_{file_meta}.png")
    # plt.savefig(
    #     f"output/denoise_plots/denoise_{baseline_method}_{file_meta}.png", dpi=300)

def create_filtering_comparison_plot(
    time: np.ndarray,
    signal: np.ndarray,
    sampling_rate: float,
):
    set_plt_defaults(12)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    

    min_time = time.min()
    seconds_from_start = time - min_time
    days_from_start = seconds_from_start / (60 * 60 * 24)

    baseline1, _ = extract_baseline_and_offset(signal, sampling_rate, "fourier", cutoff_freq=(0.001, 0.05))
    baseline2, _ = extract_baseline_and_offset(signal, sampling_rate, "butterworth", cutoff_freq=0.05)
    baseline3, _ = extract_baseline_and_offset(signal, sampling_rate, "moving_average", window_size=5000)
    baseline4, _ = extract_baseline_and_offset(signal, sampling_rate, "savgol", window_size=80)

    ax1.plot(days_from_start, signal)
    ax1.plot(days_from_start, baseline1)
    ax1.set_title("Fourier")
   
    ax1.set_ylabel("Voltage [Microvolts]")
    
    ax2.plot(days_from_start, signal)
    ax2.plot(days_from_start, baseline2)
    ax2.set_title("Butterworth")
    

    
    
    ax3.plot(days_from_start, signal)
    ax3.plot(days_from_start, baseline3)
    ax3.set_title("Moving Average")
    ax3.set_xlabel("Time [Days]")
    ax3.set_ylabel("Voltage [Microvolts]")
    
    
    ax4.plot(days_from_start, signal)
    ax4.plot(days_from_start, baseline4)
    ax4.set_title("Savitzky-Golay")
    ax4.set_xlabel("Time [Days]")

    
    plt.show()