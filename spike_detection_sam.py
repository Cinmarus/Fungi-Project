import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

class peak_analyser:
    def __init__(self, df_signal, column):  # <-- Fixed constructor
        self.df = df_signal
        self.voltage_column = df_signal.columns[column]
        self.time_column = df_signal.columns[0]
        self.time_numeric = df_signal[self.time_column]
        self.voltage = df_signal[self.voltage_column]
        self.df_peaks = pd.DataFrame()  # Empty until get_peaks is called


    def get_peaks(self, prominence=100, **kwargs):
        function = self.voltage
        height_opt_pos = np.mean(function)
        height_opt_neg = -np.mean(function)

        peaks_positive, props_pos = find_peaks(
            function, height=height_opt_pos, prominence=prominence, **kwargs
        )
        peaks_negative, props_neg = find_peaks(
            -function, height=height_opt_neg, prominence=prominence, **kwargs
        )

        def build_peaks_df(peaks, props, orientation):
            if len(peaks) == 0:
                return pd.DataFrame()
            return pd.DataFrame({
                'peak_index': peaks,
                'height': abs(props.get('peak_heights', np.nan)),
                'prominence': props.get('prominences', np.nan),
                'width': props.get('widths', np.nan),
                'width_height': abs(props.get('width_heights', np.nan)),
                'left_base': props.get('left_bases', np.nan),
                'right_base': props.get('right_bases', np.nan),
                'left_ips': props.get('left_ips', np.nan),
                'right_ips': props.get('right_ips', np.nan),
                'plateau_size': props.get('plateau_sizes', np.nan),
                'orientation': orientation
            })

        df_peaks_pos = build_peaks_df(peaks_positive, props_pos, 'Positive')
        df_peaks_neg = build_peaks_df(peaks_negative, props_neg, 'Negative')
        df_peaks = pd.concat([df_peaks_pos, df_peaks_neg], ignore_index=True)
        self.df_peaks = df_peaks.sort_values(by='peak_index').reset_index(drop=True)
        return self.df_peaks

    def filter_peaks_by_params(self, **kwargs):
        filtered = self.df_peaks
        for key, value in kwargs.items():
            if value is not None and key in filtered.columns:
                filtered = filtered[filtered[key] >= value]
            elif key == "orientation" and value is not None:
                filtered = filtered[filtered["orientation"] == value]
        self.df_peaks = filtered
        return self.df_peaks

    def graph_peaks(self):
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axs[0].plot(self.time_numeric, self.voltage, label="Original Signal")
        axs[0].set_ylabel("Voltage (µV)")
        axs[0].set_title("Original Signal with Peaks")
        axs[0].legend()

        axs[1].plot(self.time_numeric, self.voltage, alpha=0.5, label="Signal")
        if not self.df_peaks.empty and 'peak_index' in self.df_peaks.columns:
            axs[1].scatter(self.time_numeric[self.df_peaks['peak_index']],
                           self.voltage[self.df_peaks['peak_index']],
                           color="red", label="Detected Peaks")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Voltage (µV)")
        axs[1].set_title("Detected Peaks")
        axs[1].legend()
        plt.tight_layout()
        plt.show()