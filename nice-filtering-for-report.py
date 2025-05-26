import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

filepath = 'data/data.csv' #format: time[s], voltage[muV]

data = pd.read_csv(filepath)

def filter(data):
    """Take DataFrame with two columns - time[s] and voltage[muV], return numpy arrays of time and filtered signal"""
    time = data.iloc[:, 0] #time axis extraction
    voltage = data.iloc[:, 1] #signal axis extraction

    time = (time - time.iloc[0]).to_numpy() #ensure time axis starts at 0

    baseline = voltage.rolling(window=5000, min_periods=0, center=True).mean().to_numpy() #calculate 5000pt moving average as baseline

    signal = voltage.to_numpy() - baseline #remove baseline from signal

    signal = savgol_filter(signal, window_length=50,  polyorder=3) #apply third-order Savitzky-Golay Filter

    return time, signal

time, signal = filter(data)

plt.plot(time, signal)
plt.show()