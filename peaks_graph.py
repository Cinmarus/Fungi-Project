import matplotlib.pyplot as plt
import numpy as np

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


def graph_multiple_signal_with_peaks(df, time_column, signal_columns, peak_analyser_instances):

    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    plt.figure(figsize=(12, 5))
    if len(signal_columns) != len(peak_analyser_instances):
        print("Error: Number of signal columns and peak_analyser instances do not match.")
        return

    for idx, signal_column in enumerate(signal_columns):
        if signal_column not in df.columns:
            print(f"Warning: Signal column '{signal_column}' not found in DataFrame. Skipping.")
            continue

        current_peak_analyzer_instance = peak_analyser_instances[idx]
        df_peaks = current_peak_analyzer_instance.df_peaks 

        plt.plot(df[time_column], df[signal_column],
                 label=f'{signal_column}',
                 color=colors[idx % len(colors)],
                 alpha=0.7)

        if not df_peaks.empty:
            
            peak_indices = df_peaks['peak_index'].values 
            peak_times = df[time_column].iloc[peak_indices]
            peak_voltages = df[signal_column].iloc[peak_indices]
            plt.plot(peak_times, peak_voltages,
                     'x', 
                     color=colors[idx % len(colors)], 
                     markersize=8,
                     label=f'{signal_column} Peaks', 
                     linestyle='None') 


    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.title('Signals with Peaks')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Plots/multiple_signals_plot_with_peaks.png")
    plt.show()

def compare_number_of_peaks_list(methods_names, peak_analyser_instances):
    if len(methods_names) != len(peak_analyser_instances):
        print("Error: Number of method names and peak_analyser instances do not match.")
        return

    num_peaks = [len(pa.df_peaks) for pa in peak_analyser_instances]
    bar_width = 0.6
    x_positions = np.arange(len(methods_names))

    plt.figure(figsize=(10, 6)) 

    bars = plt.bar(x_positions, num_peaks, bar_width, color='skyblue')

    plt.xlabel("Filtering Method")
    plt.ylabel("Number of Detected Peaks")
    plt.title("Comparison of Number of Peaks by Filtering Method")
    plt.xticks(x_positions, methods_names)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center') 

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("Plots/peak_count_comparison_bar_plot_list.png")
    plt.show()





