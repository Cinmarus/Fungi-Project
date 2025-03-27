import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define constants
CHUNK_SIZE = 10_000
LOW_FREQ = 0.001  # [Hz]
HIGH_FREQ = 0.05  # [Hz]
FILE_PATH = "data/mycelium_data_channel1-2.csv"
WINDOW_SIZE = 250_000  # number of data points for the scrolling display


def process_chunk(chunk: pd.DataFrame, col: str):
    """
    Process a single chunk by band-pass filtering the signal using FFT.
    Only frequencies between LOW_FREQ and HIGH_FREQ remain.

    Parameters:
        chunk (pd.DataFrame): Data chunk containing "timestamp" and signal data.
        col (str): Name of the signal column.

    Returns:
        timestamps (np.ndarray): Array of timestamps.
        raw_signal (np.ndarray): Raw signal values.
        filtered_signal (np.ndarray): The filtered signal.
    """
    # Ensure that 'timestamp' is numeric and remove invalid rows.
    chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce")
    chunk = chunk.dropna(subset=["timestamp", col])

    timestamps = chunk["timestamp"].values
    raw_signal = chunk[col].values

    # Compute sampling interval assuming timestamps are in seconds.
    dt = np.median(np.diff(timestamps))
    N = len(raw_signal)

    # Perform FFT
    fft_signal = np.fft.fft(raw_signal)
    freqs = np.fft.fftfreq(N, d=dt)

    # Build a mask that retains frequencies in our band
    mask = (np.abs(freqs) >= LOW_FREQ) & (np.abs(freqs) <= HIGH_FREQ)

    # Zero out frequencies outside the band
    fft_filtered = fft_signal * mask

    # Inverse FFT to return to time domain
    filtered_signal = np.fft.ifft(fft_filtered).real
    return timestamps, raw_signal, filtered_signal


# Turn on matplotlib interactive mode for live updates.
plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))
raw_line, = ax.plot([], [], label="Raw Signal", color="blue", alpha=0.7)
filt_line, = ax.plot([], [], label="Filtered Signal",
                     linestyle="--", color="red", alpha=0.7)
ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")
ax.legend()
ax.grid(True)

# Arrays to store processed data.
all_timestamps = np.array([])
all_raw = np.array([])
all_filt = np.array([])

# Process the CSV file in chunks.
for chunk in pd.read_csv(FILE_PATH, chunksize=CHUNK_SIZE):
    # Rename the first column to 'timestamp'
    chunk = chunk.rename(columns={chunk.columns[0]: "timestamp"})

    # Assume the second column contains the signal.
    signal_column = chunk.columns[1]

    # Process the chunk to extract timestamps, raw, and filtered signals.
    timestamps, raw_signal, filtered_signal = process_chunk(
        chunk, signal_column)

    # Append the new data to the aggregated arrays.
    all_timestamps = np.concatenate((all_timestamps, timestamps))
    all_raw = np.concatenate((all_raw, raw_signal))
    all_filt = np.concatenate((all_filt, filtered_signal))

    # Create a scrolling view by only taking the last WINDOW_SIZE points.
    if len(all_timestamps) > WINDOW_SIZE:
        view_timestamps = all_timestamps[-WINDOW_SIZE:]
        view_raw = all_raw[-WINDOW_SIZE:]
        view_filt = all_filt[-WINDOW_SIZE:]
    else:
        view_timestamps = all_timestamps
        view_raw = all_raw
        view_filt = all_filt

    # Update the plot with the most recent data.
    raw_line.set_data(view_timestamps, view_raw)
    filt_line.set_data(view_timestamps, view_filt)

    # Adjust the axes limits so that the scrolling window is visible.
    ax.set_xlim(view_timestamps[0], view_timestamps[-1])
    y_min = min(np.min(view_raw), np.min(view_filt))
    y_max = max(np.max(view_raw), np.max(view_filt))
    # Add some margin on the y-axis.
    ax.set_ylim(y_min - 0.1 * abs(y_min), y_max + 0.1 * abs(y_max))

    plt.draw()
    plt.pause(0.001)  # brief pause to simulate real-time updating

plt.ioff()
plt.show()
