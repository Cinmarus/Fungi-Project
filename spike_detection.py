import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load the CSV file
file_path = "test2.csv"
df = pd.read_csv(file_path, delimiter=",", skipinitialspace=True)

# Extract time and voltage data
time_column = df.columns[0]  # First column is time
voltage_column = df.columns[1]  # Second column is the voltage signal
time = df[time_column]
voltage = df[voltage_column]

# Convert time to numerical values if needed (e.g., seconds since start)
time_numeric = np.arange(len(time))  # Assuming uniform sampling

# Detect peaks
peaks, properties = find_peaks(voltage, height=None, distance=None, prominence=None)

# Create a figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot the original signal
axs[0].plot(time_numeric, voltage, label="Original Signal")
axs[0].set_ylabel("Voltage (µV)")
axs[0].set_title("Original Signal with Peaks")
axs[0].legend()

# Plot the detected peaks
axs[1].plot(time_numeric, voltage, label="Signal", alpha=0.5)
axs[1].scatter(time_numeric[peaks], voltage[peaks], color="red", label="Detected Peaks", zorder=3)
axs[1].set_xlabel("Time (seconds)")
axs[1].set_ylabel("Voltage (µV)")
axs[1].set_title("Detected Peaks")
axs[1].legend()

plt.tight_layout()
plt.show()