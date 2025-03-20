import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy import signal


data = pd.read_csv("data/new_data.csv")

if os.path.isfile('data/data_pickled'):
    data = pd.read_pickle("data/data_pickled")
    print("unpickled")
else:
    data.to_pickle("data/data_pickled")
    print("pickled")

data["Time"] -= data["Time"].iloc[0]

totalTime = data["Time"].iloc[-1]-data["Time"].iloc[0]

Fs = len(data["Time"])/totalTime

data["Rolling Average"] = data["Voltage"].rolling(window=100, min_periods=0).mean()
data["Baseline"] = data["Voltage"].rolling(window=50000, min_periods=0).mean()
data["Flattened"] = data["Rolling Average"] - data["Baseline"]


print(data.head())

# plt.plot(data["Voltage"])
# plt.plot(data["Rolling Average"])
# plt.plot(data["Baseline"])
# plt.plot(data["Flattened"])
# plt.show()


NFFT = 1024  # the length of the windowing segments

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
ax1.plot(data["Time"], data["Voltage"])
ax1.set_ylabel('Signal')


Pxx, freq, t = matplotlib.mlab.specgram(data["Voltage"], NFFT=NFFT, noverlap=NFFT//2, Fs=Fs)

Pxx_log = 10 * np.log10(Pxx + 1e-10)
ax2.pcolormesh(t, freq, Pxx_log, cmap='plasma', shading='auto', vmin=np.percentile(Pxx_log, 1), vmax=np.percentile(Pxx_log, 99))
# ax2.set_yscale('log')

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Frequency (Hz)')


plt.xlim([data["Time"].iloc[0], data["Time"].iloc[-1]])
plt.show()
