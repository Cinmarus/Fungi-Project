from datetime import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from data_loader import load_data_from_file


data_file = "data/new_data.csv"
data = load_data_from_file(data_file)

cache_file = "valid_frames_cache.txt"

with open(cache_file, "r") as f:
    valid_files = [line.strip() for line in f if line.strip()]

video_timestamps = []

for file_path in valid_files:
    filename = os.path.splitext(os.path.basename(file_path))[0]
    parts = filename.split(" - ")

    if len(parts) == 2:
        ts_str = parts[1]
        try:
            dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
            video_timestamps.append(dt.timestamp())
        except Exception as e:
            print(f"Error parsing timestamp from {filename}: {e}")

if not video_timestamps:
    print("No valid video timestamps could be extracted.")
    exit(1)

rand_timestamp = video_timestamps[-1]

fig, ax = plt.subplots()

start_timestamp = min(video_timestamps)
end_timestamp = max(video_timestamps)

y_min = data.iloc[:, 1].min()
y_max = data.iloc[:, 1].max()

ax.grid()

ax.set_xlim(start_timestamp, end_timestamp)
ax.set_ylim(y_min, y_max)

plot_data = data.loc[data["timestamp"] < rand_timestamp]
print(plot_data.head())

ax.plot(plot_data.iloc[:, 0], plot_data.iloc[:, 1])

plt.show()
