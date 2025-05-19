import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load CSV data
df = pd.read_csv('new_data.csv', index_col=0)

df.index = pd.to_timedelta(df.index)

# Remove Baseline
df_corrected = df - df.mean()

peaks_list = []
drops_list = []

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Original Data
for column in df.columns:
    axes[0].plot(df.index.total_seconds(), df[column], label=column)
    axes[0].set_title('Original PicoLog Data')
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Microvolts (µV)')
    axes[0].legend(loc='best')
    axes[0].grid(True)

# Baseline Corrected Data
prominence_value = 70
for column in df_corrected.columns:
    axes[1].plot(df_corrected.index.total_seconds(),
                 df_corrected[column], label=column)
    # Find and plot large peaks
    peaks, _ = find_peaks(df_corrected[column], prominence=prominence_value)
    axes[1].plot(df_corrected.index.total_seconds()[peaks],
                 df_corrected[column].iloc[peaks], 'rx')
    peaks_list.extend(zip(df_corrected.index.total_seconds()[
                      peaks], df_corrected[column].iloc[peaks]))
    # Find and plot large drops
    drops, _ = find_peaks(-df_corrected[column], prominence=prominence_value)
    axes[1].plot(df_corrected.index.total_seconds()[drops],
                 df_corrected[column].iloc[drops], 'gv')
    drops_list.extend(zip(df_corrected.index.total_seconds()[
                      drops], df_corrected[column].iloc[drops]))

    axes[1].set_title(
        'Baseline Corrected PicoLog Data with Large Peaks and Drops')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Microvolts (µV, baseline corrected)')
    axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1].legend(loc='best')
    axes[1].grid(True)


def remove_duplicate_events(events, time_tolerance=0.01):
    cleaned_events = []
    events.sort()
    i = 0
    while i < len(events):
        current_event = events[i]
        duplicates = [current_event]

        j = i + 1
        while j < len(events) and abs(events[j][0] - current_event[0]) <= time_tolerance:
            duplicates.append(events[j])
            j += 1

        largest_event = max(duplicates, key=lambda x: abs(x[1]))
        cleaned_events.append(largest_event)

        i = j

    return cleaned_events


peaks_list = remove_duplicate_events(peaks_list)
drops_list = remove_duplicate_events(drops_list)

print("Detected Peaks:", peaks_list)
print("Detected Drops:", drops_list)

plt.tight_layout()
plt.show()
