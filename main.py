import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from peaks_graph import graph_peaks_bokeh
from scipy.signal import find_peaks, savgol_filter
from data_loader import load_data_from_file
from spike_detection import peak_analyser
from analysis import BaselineMethods, DenoiseMethods, denoise_signal, extract_baseline_and_offset


def main():
    file_path = "data/new_data.csv"
    print(f"Loading data from {file_path}...")

    df = load_data_from_file(file_path)
    print("Data loaded successfully!")
    print(df.head())  # Display the first few rows of the data

    print("Calculating sampling rate...")
    # Replace "timestamp" with the correct column name if different
    time = df.iloc[:, 0]
    sampling_rate = 1 / np.mean(np.diff(time))
    print(f"Sampling rate: {sampling_rate} Hz")

    # Extract baseline and offset signal
    print("Extracting baseline and offset signal...")
    try:
        baseline, offset_signal = extract_baseline_and_offset(
            # Replace "Voltage" with the correct column name if different
            df.iloc[:, 1],
            sampling_rate=sampling_rate,
            method="moving_average",  # Example method
            window_size=5000  # Example window size
        )

        print("Baseline and offset signal extracted!")
    except AttributeError as e:
        print(f"Error: {e}")
        return

    # denoise the offset signal
    denoised_offset = savgol_filter(offset_signal, 50, 3)

    # Add the offset signal to the DataFrame for further analysis
    df["Offset_Signal"] = denoised_offset
    print(df.head())  # Display the updated DataFrame

    # Continue with further analysis (e.g., spike detection, visualization, etc.)
    # Example: Perform spike detection
    print("Performing spike detection...")
    pa = peak_analyser(df, df.columns.get_loc("Offset_Signal"))
    pa.get_peaks()
    pa.filter_peaks_by_params(prominence_min=50, width_min= 10)
    graph_peaks_bokeh(pa)
    pa.compare_peaks("width")


if __name__ == "__main__":
    main()
