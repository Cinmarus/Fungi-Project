import numpy as np
import matplotlib.pyplot as plt

from analysis import constrain_data_frequencies, extract_signal_from_data
from data_loader import load_data_from_file
from visualization import visualize_data, visualize_data_baseline, visualize_data_offset

# assumed that the baseline fluctuations have frequencies lower than this
BASELINE_LOW_CUTOFF_FREQ = 0.001  # [Hz]
BASELINE_HIGH_CUTOFF_FREQ = 0.05  # [Hz]

file_path = "data/mycelium_data_channel1-2.csv"

data = load_data_from_file(file_path)
data = data.iloc[:100000]


analysis_col = data.columns[-1]

# visualize_data_baseline(analysed_data, analysis_col)
# visualize_data_offset(analysed_data, analysis_col)

col_data = data[analysis_col]
dt = np.median(np.diff(data["timestamp"].values))
freq_band_data = constrain_data_frequencies(
    col_data, BASELINE_LOW_CUTOFF_FREQ, BASELINE_HIGH_CUTOFF_FREQ, dt)

plt.plot(data["timestamp"], col_data, label="raw data")
plt.plot(data["timestamp"], freq_band_data, label="bounded freqs")

plt.legend()

plt.show()
