import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy import signal



if os.path.isfile('data/data_pickled'):
    data = pd.read_pickle("data/data_pickled")
    print("unpickled")
else:
    if os.path.isfile('data/new_data.csv'):
        data = pd.read_csv("data/new_data.csv")
        data.to_pickle("data/data_pickled")
        print("pickled")
    else: 
        print("no data file present")
        exit()

    

data["Time"] -= data["Time"].iloc[0]

data["Rolling Average"] = data["Voltage"].rolling(window=100, min_periods=0, center=True).mean()
data["Baseline"] = data["Voltage"].rolling(window=50000, min_periods=0, center=True).mean()
data["Flattened"] = data["Rolling Average"] - data["Baseline"]

time = data["Time"]
voltage = data["Voltage"]
rollingAverage = data["Rolling Average"]
baseline = data["Baseline"]
flattened = data["Flattened"]

print(data.head())

plt.plot(data["Voltage"])
plt.plot(data["Rolling Average"])
plt.plot(data["Baseline"])
plt.plot(data["Flattened"])
plt.show()


def plotSpectrogram(time, signal):

    totalTime = time.iloc[-1]-time.iloc[0]
    Fs = len(time)/totalTime    

    NFFT = 2**16  # the length of the windowing segments

    fig = plt.figure(constrained_layout=True)

    gs = fig.add_gridspec(3, 2)

    ax1 = fig.add_subplot(gs[0, 1]) #signal plot
    ax2 = fig.add_subplot(gs[1:, 1]) # spectrogram
    ax3 = fig.add_subplot(gs[1:, 0]) # fourier transform
        
    ax1.plot(time, signal)
    ax1.set_xlim([0, time.iloc[-1]])
    ax1.set_ylabel('Signal')

    Pxx, freq, t = matplotlib.mlab.specgram(signal, NFFT=NFFT, noverlap=NFFT*3//4, Fs=Fs)

    Pxx_log = 10 * np.log10(Pxx + 1e-10)
    pcm = ax2.pcolormesh(t, freq, Pxx_log, cmap='plasma', shading='auto', vmin=np.percentile(Pxx_log, 1), vmax=np.percentile(Pxx_log, 99))
    fig.colorbar(pcm, ax=ax2)
    # ax2.set_yscale('log')

    ax2.set_xlabel('Time (s)')
    # ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlim([0, time.iloc[-1]])

    yf = np.fft.rfft(signal)
    xf = np.fft.rfftfreq(signal.size, d=1/Fs)

    ax2.set_yscale('log')
    ax3.set_yscale('log')

    ax3.plot(yf, xf)
    # ax3.set_xlim([None, None])
    ax2.set_ylim(ax3.get_ylim())

    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_xlabel('Amplitude')
    ax3.set_xscale('log')

    


    # plt.xlim([x.iloc[0], x.iloc[-1]])
    plt.show()

# plotSpectrogram(time, voltage)
# plotSpectrogram(time, rollingAverage)
# plotSpectrogram(time, flattened)
# plotSpectrogram(time, baseline)