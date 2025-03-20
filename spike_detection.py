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

height_opt = - max(voltage)

# Detect peaks
peaks1, properties1 = find_peaks(voltage, height=height_opt, distance=1, prominence=0.01, width = 0.01, wlen = None, plateau_size=0.1, rel_height=0.1, threshold=0)
peaks2, properties2 = find_peaks(-voltage, height=height_opt, distance=1, prominence=0.01, width = 0.01, wlen = None, plateau_size=0.1, rel_height=0.1, threshold=0)

print(properties1)

peaks1_data = {
    'peak_index': peaks1,
    'height': properties1['peak_heights'] if 'peak_heights' in properties1 else None,
    'prominence': properties1['prominences'] if 'prominences' in properties1 else None,
    'width': properties1['widths'] if 'widths' in properties1 else None,
    'width_height': properties1['width_heights'] if 'width_heights' in properties1 else None,
    'left_base': properties1['left_bases'] if 'left_bases' in properties1 else None,
    'right_base': properties1['right_bases'] if 'right_bases' in properties1 else None,
    'left_ips': properties1['left_ips'] if 'left_ips' in properties1 else None,
    'right_ips': properties1['right_ips'] if 'right_ips' in properties1 else None,
    'left_threshold': properties1['left_thresholds'] if 'left_thresholds' in properties1 else None,
    'right_threshold': properties1['right_thresholds'] if 'right_thresholds' in properties1 else None,
    'wlen': properties1['wlen'] if 'wlen' in properties1 else None,
    'rel_height': properties1['rel_height'] if 'rel_height' in properties1 else None,
    'plateau_size': properties1['plateau_sizes'] if 'plateau_sizes' in properties1 else None
}

peaks2_data = {
    'peak_index': peaks2,
    'height': properties2['peak_heights'] if 'peak_heights' in properties2 else None,
    'prominence': properties2['prominences'] if 'prominences' in properties2 else None,
    'width': properties2['widths'] if 'widths' in properties2 else None,
    'width_height': properties2['width_heights'] if 'width_heights' in properties2 else None,
    'left_base': properties2['left_bases'] if 'left_bases' in properties2 else None,
    'right_base': properties2['right_bases'] if 'right_bases' in properties2 else None,
    'left_ips': properties2['left_ips'] if 'left_ips' in properties2 else None,
    'right_ips': properties2['right_ips'] if 'right_ips' in properties2 else None,
    'left_threshold': properties2['left_thresholds'] if 'left_thresholds' in properties2 else None,
    'right_threshold': properties2['right_thresholds'] if 'right_thresholds' in properties2 else None,
    'wlen': properties2['wlen'] if 'wlen' in properties2 else None,
    'rel_height': properties2['rel_height'] if 'rel_height' in properties2 else None,
    'plateau_size': properties2['plateau_sizes'] if 'plateau_sizes' in properties2 else None
}


df_peaks1 = pd.DataFrame(peaks1_data)
df_peaks2 = pd.DataFrame(peaks2_data)

df_peaks = pd.concat([df_peaks1, df_peaks2], ignore_index=True)

df_peaks = df_peaks.sort_values(by='peak_index').reset_index(drop=True)
peaks = df_peaks['peak_index']
print(df_peaks)

average_amplitude = np.mean(np.abs(voltage[peaks]))
print(f"Average Amplitude of Peaks: {average_amplitude:.4f} µV")

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