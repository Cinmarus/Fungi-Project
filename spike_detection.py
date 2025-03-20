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
peaks1, properties1 = find_peaks(voltage, height=1, distance=1, prominence=1, width = 1)
peaks2, properties2 = find_peaks(-voltage, height=1, distance=1, prominence=1, width = 1)

print(properties1)

peaks1_data = {
    'peak_index': peaks1,
    'height': properties1['peak_heights'],
    'prominence': properties1['prominences'],
    'width': properties1['widths'] if 'widths' in properties1 else None
}

peaks2_data = {
    'peak_index': peaks2,
    'height': properties2['peak_heights'],
    'prominence': properties2['prominences'],
    'width': properties2['widths'] if 'widths' in properties1 else None
}


df_peaks1 = pd.DataFrame(peaks1_data)
df_peaks2 = pd.DataFrame(peaks2_data)

df_peaks = pd.concat([df_peaks1, df_peaks2], ignore_index=True)

df_peaks = df_peaks.sort_values(by='peak_index').reset_index(drop=True)
peaks = df_peaks['peak_index']
print(df_peaks)



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