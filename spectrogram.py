import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import analysis

import Data_visualisation_function
Data_visualisation_function.set_plt_defaults()

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

data["Rolling Average"] = data["Voltage"].rolling(window=50, min_periods=0, center=True).mean()
data["Baseline"] = data["Voltage"].rolling(window=5000, min_periods=0, center=True).mean()
data["Flattened"] = data["Voltage"] - data["Baseline"]

interestingFrequencies = [0.01666, 0.01993, 0.02327]

def multipleFourier(signal, frequencies, width):
    signal = np.array(signal)
    newSignal = signal - signal
    for freq in frequencies:
        newSignal += analysis.extract_baseline_and_offset(signal, 100/6, 'fourier', (freq-width/2, freq+width/2))[0]
    return newSignal

time = data["Time"]
voltage = data["Voltage"]
rollingAverage = data["Rolling Average"]
baseline = data["Baseline"]
flattened = data["Flattened"]
butterworth = analysis.extract_baseline_and_offset(voltage, 100/6, 'butterworth', cutoff_freq=0.001)[1]
fourier = analysis.extract_baseline_and_offset(np.array(voltage), 100/6, 'fourier', (0.001, 0.05))[0]
multiFourier = multipleFourier(voltage, frequencies=interestingFrequencies, width=0.00035)


print(data.head())

# plt.plot(data["Voltage"])
# plt.plot(data["Rolling Average"])
# plt.plot(data["Baseline"])
# plt.plot(data["Flattened"])
# plt.show()



def plotSpectrogram(time, signal, title, save=False, filename=None, log=False):

    totalTime = time.iloc[-1]-time.iloc[0]
    Fs = len(time)/totalTime    

    NFFT = 2**16  # length of the windowing segments

    fig = plt.figure(constrained_layout=True)

    gs = fig.add_gridspec(3, 3)

    ax1 = fig.add_subplot(gs[0, 1:]) #signal plot
    ax2 = fig.add_subplot(gs[1:, 1:]) # spectrogram
    ax3 = fig.add_subplot(gs[1:, 0]) # fourier transform

    ax1.plot(time, signal)
    ax1.set_xlim([0, time.iloc[-1]])
    ax1.set_ylabel(r'Voltage [$\mu$V]')

    Pxx, freq, t = matplotlib.mlab.specgram(signal, NFFT=NFFT, noverlap=NFFT*3//4, Fs=Fs)

    Pxx_log = 10 * np.log10(Pxx + 1e-10) # logarithmic amplitude scale
    pcm = ax2.pcolormesh(t, freq, Pxx_log, cmap='plasma', shading='auto', vmin=np.percentile(Pxx_log, 0.1), vmax=np.percentile(Pxx_log, 99.9))
    cb = fig.colorbar(pcm, ax=ax2)
    cb.set_label('Amplitude (log scale)')
    ax2.set_xlabel('Time [s]')
    ax2.set_xlim([0, time.iloc[-1]])

    yf = np.fft.rfft(signal)
    xf = np.fft.rfftfreq(signal.size, d=1/Fs)

    ax3.plot(np.abs(yf), xf)
    ax3.set_xlim([1e3, None])

    ax3.set_xlabel('Amplitude (log scale)')
    ax3.set_xscale('log')

    if log:
        ax3.set_ylabel('Frequency [Hz] (log scale)')
        ax2.set_yscale('log')
        ax3.set_yscale('log')
        ax3.set_ylim([1e-4, 8+1/3])
        ax2.set_ylim(ax3.get_ylim())
    else:
        ax3.set_ylabel('Frequency [Hz]')
        ax3.set_ylim(0, 8+1/3)
        ax2.set_ylim(ax3.get_ylim())

    ax1.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    # plt.tight_layout()
    if not save:
        plt.show()
    elif save:
        plt.savefig(str(filename))
        plt.close(fig)
        

def saveSpectrogram(time, signal, filename, log=False, NFFT=2**16, zoomToInterestingFrequencies=False):
    totalTime = time.iloc[-1]-time.iloc[0]
    Fs = len(time)/totalTime    

    # NFFT = 2**18  # length of the windowing segments

    fig, ax = plt.subplots()


    Pxx, freq, t = matplotlib.mlab.specgram(signal, NFFT=NFFT, noverlap=NFFT*3//4, Fs=Fs)

    Pxx_log = 10 * np.log10(Pxx + 1e-10) # logarithmic amplitude scale
    pcm = ax.pcolormesh(t, freq, Pxx_log, cmap='plasma', shading='auto', vmin=np.percentile(Pxx_log, 0.1), vmax=np.percentile(Pxx_log, 99.9))
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label('Amplitude (log scale)')
    ax.set_xlabel('Time [s]')
    if log:
        ax.set_ylabel('Frequency [Hz] (log scale)')
        ax.set_yscale('log')
        ax.set_ylim(1e-4, 100/12)
    else:
        ax.set_ylabel('Frequency [Hz]')
    if zoomToInterestingFrequencies:
        xlow = min(interestingFrequencies) - 0.4 * min(interestingFrequencies)
        xhigh = max(interestingFrequencies) + 0.4 * max(interestingFrequencies)
        ax.set_ylim(xlow, xhigh)
    ax.set_xlim([0, time.iloc[-1]])

    plt.tight_layout()
    plt.savefig(str(filename))
    plt.close(fig)

def saveSignal(time, signal, filename):
    fig, ax = plt.subplots()
    ax.plot(time, signal)
    ax.set_xlim([0, time.iloc[-1]])
    ax.set_ylabel(r'Voltage [$\mu$V]')
    ax.set_xlabel("Time [s]")

    plt.savefig(str(filename))
    plt.close(fig)

def linLogSpectrograms(time, signal, folderName, signalName, extension, addition='', NFFT=2**16, big=False):
    if big:
        print("Saving Linear Scale Spectrogram...")
        plotSpectrogram(time, signal, signalName, save=True, filename=folderName + signalName + 'BigSpectrogramLinScale' + addition + extension, log=False)
        print("Saving Logarithmic Spectrogram...")
        plotSpectrogram(time, signal, signalName, save=True, filename=folderName + signalName + 'BigSpectrogramLogScale' + addition + extension, log=True)
    else:
        print("Saving Linear Scale Spectrogram...")
        saveSpectrogram(time, signal, folderName + signalName + 'SpectrogramLinScale' + addition + extension, log=False, NFFT=NFFT)
        print("Saving Logarithmic Spectrogram...")
        saveSpectrogram(time, signal, folderName + signalName + 'SpectrogramLogScale' + addition + extension, log=True, NFFT=NFFT)

def saveSpectrogramSet(time, signal, signalName, folderName='plots', extension='.png', changes=False, big=False):
    print('Starting ' + signalName)
    if not folderName[-1] == '/':
        folderName += '/'
    print("Saving Signal...")
    saveSignal(time, signal, folderName + signalName + extension)
    if not big:
        linLogSpectrograms(time, signal, folderName, signalName, extension)
    else: 
        linLogSpectrograms(time, signal, folderName, signalName, extension, big=big)
    if changes:
        for change in changes:
            match change[0]:
                case 'NFFT':
                    for i in change[1]:
                        linLogSpectrograms(time, signal, folderName, signalName, extension, NFFT=i, addition='(NFFT=' + str(i) + ')')




# plotSpectrogram(time, voltage, "Voltage")
# plotSpectrogram(time, rollingAverage, "Rolling Average Spectrogram")
# plotSpectrogram(time, flattened, "Flattened Data")
# plotSpectrogram(time, fourier, 'Fourier')
# plotSpectrogram(time, multiFourier, "MultiFourier")

saveSpectrogramSet(time, signal=voltage, signalName='Voltage', changes=[['NFFT', [2**16, 2**18]]])
saveSpectrogramSet(time, flattened, '5000ptFlattened')
saveSpectrogramSet(time, butterworth, 'Butterworth')
saveSpectrogramSet(time, rollingAverage, "50PtRolling")
saveSpectrogramSet(time, fourier, 'Fourier')
saveSpectrogramSet(time, multiFourier, 'MultiFourier')

# saveSpectrogram(time, multiFourier, 'plots/MultiFourierSpectrogramLogZoom', log=True, zoomToInterestingFrequencies=True)
# saveSpectrogramSet(time, voltage, 'Voltage', big=True)