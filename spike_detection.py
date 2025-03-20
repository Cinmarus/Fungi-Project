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
peaks_positive, properties_positive = find_peaks(voltage, height=height_opt, distance=1, prominence=0.01, width = 0.01, wlen = None, plateau_size=0.1, rel_height=0.1, threshold=0)
peaks_negative, properties_negative = find_peaks(-voltage, height=height_opt, distance=1, prominence=0.01, width = 0.01, wlen = None, plateau_size=0.1, rel_height=0.1, threshold=0)

print(properties_positive)

peaks_positive_data = {
    'peak_index': peaks_positive,
    'height': properties_positive['peak_heights'] if 'peak_heights' in properties_positive else None,
    'prominence': properties_positive['prominences'] if 'prominences' in properties_positive else None,
    'width': properties_positive['widths'] if 'widths' in properties_positive else None,
    'width_height': properties_positive['width_heights'] if 'width_heights' in properties_positive else None,
    'left_base': properties_positive['left_bases'] if 'left_bases' in properties_positive else None,
    'right_base': properties_positive['right_bases'] if 'right_bases' in properties_positive else None,
    'left_ips': properties_positive['left_ips'] if 'left_ips' in properties_positive else None,
    'right_ips': properties_positive['right_ips'] if 'right_ips' in properties_positive else None,
    'left_threshold': properties_positive['left_thresholds'] if 'left_thresholds' in properties_positive else None,
    'right_threshold': properties_positive['right_thresholds'] if 'right_thresholds' in properties_positive else None,
    'wlen': properties_positive['wlen'] if 'wlen' in properties_positive else None,
    'rel_height': properties_positive['rel_height'] if 'rel_height' in properties_positive else None,
    'plateau_size': properties_positive['plateau_sizes'] if 'plateau_sizes' in properties_positive else None
}

peaks_negative_data = {
    'peak_index': peaks_negative,
    'height': properties_negative['peak_heights'] if 'peak_heights' in properties_negative else None,
    'prominence': properties_negative['prominences'] if 'prominences' in properties_negative else None,
    'width': properties_negative['widths'] if 'widths' in properties_negative else None,
    'width_height': properties_negative['width_heights'] if 'width_heights' in properties_negative else None,
    'left_base': properties_negative['left_bases'] if 'left_bases' in properties_negative else None,
    'right_base': properties_negative['right_bases'] if 'right_bases' in properties_negative else None,
    'left_ips': properties_negative['left_ips'] if 'left_ips' in properties_negative else None,
    'right_ips': properties_negative['right_ips'] if 'right_ips' in properties_negative else None,
    'left_threshold': properties_negative['left_thresholds'] if 'left_thresholds' in properties_negative else None,
    'right_threshold': properties_negative['right_thresholds'] if 'right_thresholds' in properties_negative else None,
    'wlen': properties_negative['wlen'] if 'wlen' in properties_negative else None,
    'rel_height': properties_negative['rel_height'] if 'rel_height' in properties_negative else None,
    'plateau_size': properties_negative['plateau_sizes'] if 'plateau_sizes' in properties_negative else None
}


df_peaks_positive = pd.DataFrame(peaks_positive_data)
df_peaks_negative = pd.DataFrame(peaks_negative_data)

df_peak_complete = pd.concat([df_peaks_positive, df_peaks_negative], ignore_index=True)

df_peak_complete = df_peak_complete.sort_values(by='peak_index').reset_index(drop=True)
peaks_complete = df_peak_complete['peak_index']
print(df_peak_complete)

average_amplitude = np.mean(np.abs(voltage[peaks_complete]))
print(f"Average Amplitude of Peaks: {average_amplitude:.4f} µV")

time_differences = np.diff(time_numeric[peaks_complete])
average_frequency = 1 / np.mean(time_differences)
print(f"Average frequency of Peaks: {average_frequency:.4f} Hz")

# Create a figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot the original signal
axs[0].plot(time_numeric, voltage, label="Original Signal")
axs[0].set_ylabel("Voltage (µV)")
axs[0].set_title("Original Signal with Peaks")
axs[0].legend()

# Plot the detected peaks
axs[1].plot(time_numeric, voltage, label="Signal", alpha=0.5)
axs[1].scatter(time_numeric[peaks_complete], voltage[peaks_complete], color="red", label="Detected Peaks", zorder=3)
axs[1].set_xlabel("Time (seconds)")
axs[1].set_ylabel("Voltage (µV)")
axs[1].set_title("Detected Peaks")
axs[1].legend()

plt.tight_layout()
plt.show()