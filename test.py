import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# test = pd.read_csv("data/data_unshielded.csv")
# test.to_pickle("df")

test = pd.read_pickle("df")
print(test.head())

channels = [test["Ch_1"], test["Ch_2"], test["Ch_3"], test["Ch_4"]]

transformedChannels = []
for channel in channels:
    transformedChannels.append(np.fft.fft(channel))

sampleTime = 60 #ms
dataGap = 60/1000 #s

xs = np.fft.fftfreq(len(transformedChannels[0]), d=dataGap)


sine_freq = 10  # Choose your desired frequency in Hz
t = np.arange(0, len(transformedChannels[0]) * dataGap, dataGap)  # Time vector
sine_wave = np.sin(2 * np.pi * sine_freq * t)*4  # Generate sine wave

# Compute FFT of sine wave
sine_fft = np.fft.fft(sine_wave)

for channel in transformedChannels:
    plt.plot(xs, np.abs(channel))

plt.plot(xs, np.abs(sine_fft), 'k--', label=f"Sine {sine_freq} Hz", linewidth=2)

plt.show()