import scipy 
from scipy.signal import savgol_filter
from data_loader import load_data_from_file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = "real_test.csv"

raw_df = load_data_from_file(file_path)
df = raw_df

analysis_col = df.columns[1]

data = df[analysis_col]
data = np.nan_to_num(data, nan=0.0)

time = df["timestamp"]

dt = np.median(np.diff(time.values))  # [s]
sampling_rate = 1 / dt  # [Hz]

denoised = savgol_filter(data, 17, 3)

plt.plot(data)
plt.plot(denoised)
plt.xlabel('Time')
plt.ylabel("Voltage")
plt.legend()
plt.show()
