import matplotlib.pyplot as plt

def graph_peaks(pa):
    time = pa.time_numeric
    voltage = pa.voltage
    peak_indices = pa.df_peaks['peak_index']

    plt.figure(figsize=(12, 5))
    plt.plot(time, voltage, label='Signal')
    plt.plot(time[peak_indices], voltage[peak_indices], 'ro', label='Peaks')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.title('Detected Peaks')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Plots/peaks_plot.png")
    plt.show()


def graph_multiple_signal(df, time_column, signal_columns):
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    plt.figure(figsize=(12, 5))

    for idx, signal_column in enumerate(signal_columns):
        plt.plot(df[time_column], df[signal_column], 
                 label=signal_column, 
                 color=colors[idx % len(colors)])

    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.title('Comparison of Filtered Signals')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Plots/multiple_signals_plot.png")
    plt.show()


