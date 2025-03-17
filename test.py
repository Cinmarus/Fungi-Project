import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os


# matplotlib.use('Agg')
plt.rcParams['agg.path.chunksize'] = 100000000
# plt.rcParams['path.simplify'] = True
# plt.rcParams['path.simplify_threshold'] = 0.1


data = pd.read_csv("data/new_data.csv")

if os.path.isfile('data/data_pickled'):
    data = pd.read_pickle("data/data_pickled")
    print("unpickled")
else:
    data.to_pickle("data/data_pickled")
    print("pickled")







data["Rolling Average"] = data["Voltage"].rolling(window=100, min_periods=0).mean()
data["Baseline"] = data["Voltage"].rolling(window=50000, min_periods=0).mean()
data["Flattened"] = data["Rolling Average"] - data["Baseline"]


print(data.head())

# plt.plot(data["Voltage"])
# plt.plot(data["Rolling Average"])
# plt.plot(data["Baseline"])
# plt.plot(data["Flattened"])
# plt.show()


NFFT = 65536  # the length of the windowing segments
Fs = 100/6  # the sampling frequency

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
ax1.plot(data["Voltage"])
ax1.set_ylabel('Signal')

Pxx, freqs, bins, im = ax2.specgram(data["Voltage"], NFFT=NFFT, noverlap=NFFT//2, Fs=Fs)
# The `specgram` method returns 4 objects. They are:
# - Pxx: the periodogram
# - freqs: the frequency vector
# - bins: the centers of the time bins
# - im: the .image.AxesImage instance representing the data in the plot
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Frequency (Hz)')

plt.show()