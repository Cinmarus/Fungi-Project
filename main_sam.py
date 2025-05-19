import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from data_loader import load_data_from_file
from spike_detection_sam import peak_analyser
from analysis import BaselineMethods, DenoiseMethods, denoise_signal, extract_baseline_and_offset

def main():
    file_path = "data/new_data.csv"
    print(f"Loading data from {file_path}...")

    df = load_data_from_file(file_path)
    print("Data loaded successfully!")
    print(df.head())

    print("Calculating sampling rate...")
    time = df.iloc[:, 0]
    sampling_rate = 1 / np.mean(np.diff(time))
    print(f"Sampling rate: {sampling_rate:.4f} Hz")

    print("Extracting baseline and offset signal...")
    try:
        baseline, offset_signal = extract_baseline_and_offset(
            df.iloc[:, 1],
            sampling_rate=sampling_rate,
            method="moving_average",
            window_size=5000
        )
        print("Baseline and offset signal extracted!")
    except AttributeError as e:
        print(f"Error: {e}")
        return

    denoised_offset = savgol_filter(offset_signal, 50, 3)
    df["Offset_Signal"] = denoised_offset
    print(df.head())

    # Compare prominence values in a 2x2 grid
    prominence_levels = [55, 58, 60, 65]
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()

    for i, prominence in enumerate(prominence_levels):
        print(f"\nProcessing Prominence >= {prominence}...")
        pa = peak_analyser(df.copy(), df.columns.get_loc("Offset_Signal"))
        df_peaks = pa.get_peaks(prominence=prominence)

        ax = axs[i]
        ax.plot(pa.time_numeric, pa.voltage, label="Signal", alpha=0.7)

        if not df_peaks.empty and 'peak_index' in df_peaks.columns:
            ax.scatter(pa.time_numeric[df_peaks['peak_index']],
                       pa.voltage[df_peaks['peak_index']],
                       color='red', s=10, label="Peaks")
            print(f"Detected {len(df_peaks)} peaks.")
        else:
            print("No peaks detected.")

        ax.set_title(f"Prominence ≥ {prominence}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (µV)")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


main()