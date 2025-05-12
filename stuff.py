import numpy as np

from analysis import BaselineMethods, DenoiseMethods
from data_loader import load_data_from_file
from visualization import create_denoised_plot, create_filtering_comparison_plot
from visualization import visualize_data


BASE_LOW_FREQ = 0.005  # [Hz]
BASE_HIGH_FREQ = 0.001  # [Hz]

NOISE_CUTOFF_FREQ = 0.1  # [Hz]

# "fourier" or "butterworth" or "moving_average"
BASELINE_METHOD: BaselineMethods = "moving_average"
DENOISE_METHOD: DenoiseMethods = "lowpass"

WIND0W_SIZE = 500  # [-] number of samples in moving average window

file_path = "new_data.csv"

raw_df = load_data_from_file(file_path)
df = raw_df

analysis_col = df.columns[1]

data = df[analysis_col]
data = np.nan_to_num(data, nan=0.0)

time = df["timestamp"]

dt = np.median(np.diff(time.values))  # [s]
sampling_rate = 1 / dt  # [Hz]

# cutoff_band = (LOW_FREQ, HIGH_FREQ)
cutoff_band = BASE_HIGH_FREQ

# visualize_data(df)
# create_denoised_plot(
#     time,
#     data,
#     sampling_rate,
#     BASELINE_METHOD,
#     DENOISE_METHOD,
#     cutoff_band,
#     NOISE_CUTOFF_FREQ,
#     WIND0W_SIZE
# )

create_filtering_comparison_plot(time, data, sampling_rate)
