import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from peaks_graph import graph_peaks_bokeh
from scipy.signal import find_peaks
from data_loader import load_data_from_file

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', None)


file_path = "data/signal_without_baseline.csv"
df = load_data_from_file(file_path)

class peak_analyser:
    def __init__(self, df_signal, column):
        self.df = df_signal
        self.voltage_column = df_signal.columns[column]
        self.time_column = df_signal.columns[0]
        self.time_numeric = df_signal[self.time_column]
        self.voltage = df_signal[self.voltage_column]
        self.df_peaks = self.get_peaks()

    def get_peaks(self):
        function = self.voltage
        height_opt = - max(function)

        peaks_positive, properties_positive = find_peaks(function, height=height_opt, distance=1, prominence=0.01, width = 0.01, wlen = None, plateau_size=0.1, rel_height=0.1, threshold=0)
        peaks_negative, properties_negative = find_peaks(-function, height=height_opt, distance=1, prominence=0.01, width = 0.01, wlen = None, plateau_size=0.1, rel_height=0.1, threshold=0)


        peaks1_data = {
            'peak_index': peaks_positive,
            'height': abs(properties_positive['peak_heights']) if 'peak_heights' in properties_positive else None,
            'prominence': properties_positive['prominences'] if 'prominences' in properties_positive else None,
            'width': properties_positive['widths'] if 'widths' in properties_positive else None,
            'width_height': abs(properties_positive['width_heights']) if 'width_heights' in properties_positive else None,
            'left_base': properties_positive['left_bases'] if 'left_bases' in properties_positive else None,
            'right_base': properties_positive['right_bases'] if 'right_bases' in properties_positive else None,
            'left_ips': properties_positive['left_ips'] if 'left_ips' in properties_positive else None,
            'right_ips': properties_positive['right_ips'] if 'right_ips' in properties_positive else None,
            'left_threshold': properties_positive['left_thresholds'] if 'left_thresholds' in properties_positive else None,
            'right_threshold': properties_positive['right_thresholds'] if 'right_thresholds' in properties_positive else None,
            'plateau_size': properties_positive['plateau_sizes'] if 'plateau_sizes' in properties_positive else None,
            'orientation': 'Positive'
        }

        peaks2_data = {
            'peak_index': peaks_negative,
            'height': abs(properties_negative['peak_heights']) if 'peak_heights' in properties_negative else None,
            'prominence': properties_negative['prominences'] if 'prominences' in properties_negative else None,
            'width': properties_negative['widths'] if 'widths' in properties_negative else None,
            'width_height': abs(properties_negative['width_heights']) if 'width_heights' in properties_negative else None,
            'left_base': properties_negative['left_bases'] if 'left_bases' in properties_negative else None,
            'right_base': properties_negative['right_bases'] if 'right_bases' in properties_negative else None,
            'left_ips': properties_negative['left_ips'] if 'left_ips' in properties_negative else None,
            'right_ips': properties_negative['right_ips'] if 'right_ips' in properties_negative else None,
            'left_threshold': properties_negative['left_thresholds'] if 'left_thresholds' in properties_negative else None,
            'right_threshold': properties_negative['right_thresholds'] if 'right_thresholds' in properties_negative else None,
            'plateau_size': properties_negative['plateau_sizes'] if 'plateau_sizes' in properties_negative else None,
            'orientation': 'Negative'
        }


        df_peaks1 = pd.DataFrame(peaks1_data)
        df_peaks2 = pd.DataFrame(peaks2_data)
        df_peaks = pd.concat([df_peaks1, df_peaks2], ignore_index=True)
        df_peaks = df_peaks.sort_values(by='peak_index').reset_index(drop=True)
        #print(df_peaks)
        return df_peaks
    
    def filter_peaks_by_params(self, height_min=None, prominence_min=None, width_min=None, 
                            width_height_min=None, left_base_min=None, right_base_min=None, 
                            left_ips_min=None, right_ips_min=None, left_threshold_min=None,
                            right_threshold_min=None, plateau_size_min=None, orientation=None):
       
        filtered_peaks = self.df_peaks

        if height_min is not None:
            filtered_peaks = filtered_peaks[filtered_peaks['height'] >= height_min]
        if prominence_min is not None:
            filtered_peaks = filtered_peaks[filtered_peaks['prominence'] >= prominence_min]
        if width_min is not None:
            filtered_peaks = filtered_peaks[filtered_peaks['width'] >= width_min]
        if width_height_min is not None:
            filtered_peaks = filtered_peaks[filtered_peaks['width_height'] >= width_height_min]
        if left_base_min is not None:
            filtered_peaks = filtered_peaks[filtered_peaks['left_base'] >= left_base_min]
        if right_base_min is not None:
            filtered_peaks = filtered_peaks[filtered_peaks['right_base'] >= right_base_min]
        if left_ips_min is not None:
            filtered_peaks = filtered_peaks[filtered_peaks['left_ips'] >= left_ips_min]
        if right_ips_min is not None:
            filtered_peaks = filtered_peaks[filtered_peaks['right_ips'] >= right_ips_min]
        if left_threshold_min is not None:
            filtered_peaks = filtered_peaks[filtered_peaks['left_threshold'] >= left_threshold_min]
        if right_threshold_min is not None:
            filtered_peaks = filtered_peaks[filtered_peaks['right_threshold'] >= right_threshold_min]
        if plateau_size_min is not None:
            filtered_peaks = filtered_peaks[filtered_peaks['plateau_size'] >= plateau_size_min]
        if orientation is not None:
            filtered_peaks = filtered_peaks[filtered_peaks['orientation'] == orientation]

        self.df_peaks = filtered_peaks  
        return self.df_peaks

    def compare_peaks(self, parameter):
        bin = 10
        if parameter not in self.df_peaks.columns:
            print(f"Parameter '{parameter}' not found in DataFrame columns.")
            return
        if parameter == 'orientation':
            bin = 2
        
        parameter_data = self.df_peaks[parameter]
        plt.figure(figsize=(10, 6))
        plt.hist(parameter_data, bins=bin, color='skyblue', edgecolor='black', alpha=0.7)

        plt.xlabel(f"{parameter.capitalize()} Value")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of Peak {parameter.capitalize()}")

        plt.tight_layout()
        plt.show()

    def get_average_amplitude(self):
        average_amplitude = np.mean(np.abs(self.voltage[self.df_peaks['peak_index']]))
        print(f"Average Amplitude of Peaks: {average_amplitude:.4f} µV")

    def get_average_freq(self):
        time_differences = np.diff(self.time_numeric[self.df_peaks['peak_index']])
        average_frequency = 1 / np.mean(time_differences)
        print(f"Average frequency of Peaks: {average_frequency:.4f} Hz")

    def graph_peaks(self):
        # Create a figure with two subplots
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot the original signal
        axs[0].plot(self.time_numeric, self.voltage, label="Original Signal")
        axs[0].set_ylabel("Voltage (µV)")
        axs[0].set_title("Original Signal with Peaks")
        axs[0].legend()

        # Plot the detected peaks
        axs[1].plot(self.time_numeric, self.voltage, label="Signal", alpha=0.5)
        axs[1].scatter(self.time_numeric[self.df_peaks['peak_index']], self.voltage[self.df_peaks['peak_index']], color="red", label="Detected Peaks", zorder=3)
        axs[1].set_xlabel("Time (seconds)")
        axs[1].set_ylabel("Voltage (µV)")
        axs[1].set_title("Detected Peaks")
        axs[1].legend()

        plt.tight_layout()
        plt.show()



pa = peak_analyser(df, 1)
pa.get_peaks()
pa.filter_peaks_by_params(prominence_min=50)
graph_peaks_bokeh(pa)
pa.compare_peaks('prominence')