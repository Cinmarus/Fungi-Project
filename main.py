import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from peaks_graph import graph_peaks, graph_multiple_signal, graph_multiple_signal_with_peaks
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
    #df = df.iloc[:13000000, :]

    print("Calculating sampling rate...")
    # Replace "timestamp" with the correct column name if different
    time = df.iloc[:, 0]
    sampling_rate = 1 / np.mean(np.diff(time))
    print(f"Sampling rate: {sampling_rate} Hz")

    # Extract baseline and offset signal
    print("Extracting baseline and offset signal...")
    try:
        baseline_fourier, offset_signal_fourier = extract_baseline_and_offset(
            # Replace "Voltage" with the correct column name if different
            df.iloc[:, 1].to_numpy(),
            sampling_rate=sampling_rate,
            method="fourier",  # Example method
            window_size=500,  # Example window size
            cutoff_freq=(0.005, 3)  # Example cutoff frequency
        )

        print("Baseline and offset signal extracted!")
    except AttributeError as e:
        print(f"Error: {e}")
        return

    try:
        baseline_butterworth, offset_signal_butterworth = extract_baseline_and_offset(
            # Replace "Voltage" with the correct column name if different
            df.iloc[:, 1].to_numpy(),
            sampling_rate=sampling_rate,
            method="butterworth",  # Example method
            window_size=500,  # Example window size
            cutoff_freq=(0.005, 3)  # Example cutoff frequency
        )

        print("Baseline and offset signal extracted!")
    except AttributeError as e:
        print(f"Error: {e}")
        return
    
    try:
        baseline_mov_avg, offset_signal_mov_avg = extract_baseline_and_offset(
            # Replace "Voltage" with the correct column name if different
            df.iloc[:, 1].to_numpy(),
            sampling_rate=sampling_rate,
            method="moving_average",  # Example method
            window_size=500,  # Example window size
            cutoff_freq=(0.005, 3)  # Example cutoff frequency
        )

        print("Baseline and offset signal extracted!")
    except AttributeError as e:
        print(f"Error: {e}")
        return

    # denoise the offset signal
    original_signal = df.iloc[:, 1]
    offset_savgol = savgol_filter(original_signal, 50, 3)

    # Add the offset signal to the DataFrame for further analysis
    df["savgol"] = offset_savgol - baseline_mov_avg
    df["fourier"] = offset_signal_fourier - baseline_mov_avg
    df["butterworth"] = offset_signal_butterworth - baseline_mov_avg
    df["moving_average"] = original_signal - baseline_mov_avg

    # Continue with further analysis (e.g., spike detection, visualization, etc.)
    # Example: Perform spike detection
    print("Performing spike detection...")
    pa_moving_avg = peak_analyser(df, df.columns.get_loc("moving_average"))
    pa_fourier = peak_analyser(df, df.columns.get_loc("fourier"))
    pa_butterworth = peak_analyser(df, df.columns.get_loc("butterworth"))
    pa_savgol = peak_analyser(df, df.columns.get_loc("savgol"))
    pa_moving_avg.get_peaks()
    pa_fourier.get_peaks()
    pa_butterworth.get_peaks()
    pa_savgol.get_peaks()
    
    #graph_peaks(pa)
    #pa.compare_peaks("width")

    #graph_multiple_signal(df, time_column="timestamp", signal_columns=["moving_average", "butterworth", "fourier","savgol"])
    graph_multiple_signal_with_peaks(
        df,
        time_column="timestamp", 
        signal_columns=["moving_average", "butterworth", "fourier","savgol"],
        peak_analyser_instances=[pa_moving_avg, pa_butterworth, pa_fourier, pa_savgol]
    )


if __name__ == "__main__":
    main()
