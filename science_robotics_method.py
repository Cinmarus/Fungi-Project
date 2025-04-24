import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter, peak_widths
import matplotlib.pyplot as plt
from scipy.stats import norm
import csv
import statistics


class SignalPreProcessing:
    def __init__(self):
        self.crop_cnt = 0
        self.crop_interval = 600  # 1 minute intervals
        self.peaks_data = []
        self.neg_peaks_data = []
        self.p_properties = []
        self.n_properties = []
        self.p_width = []
        self.n_width = []
        self.prominence = 10
        self.rel_height = 0.4
        self.x_axis_data = []
        self.negative_x_axis_data = []
        self.width_pixel = 640
        self.height_pixel = 480
        self.max_data = 0
        self.min_data = 0
        self.fre_pos_sig = []
        self.fre_neg_sig = []
        self.box_tune = 0.2
        self.time_freq = 60
        self.total_freq = []
        self.baseline = 30
        self.neg_baseline = 10000

    def load_data(self, file, col):
        """
        Load and preprocess data from a CSV file.
        """
        data_f = []
        with open(file) as f:
            csv_f = csv.reader(f)
            print("File:", file)
            for row in csv_f:
                if row[col]:
                    try:
                        data_f.append(float(row[col]))
                    except Exception as e:
                        continue

        data_f = np.array(data_f[1::])
        target_sig = savgol_filter(data_f, 11, 3)  # Smoothing filter
        plt.plot(target_sig)
        plt.xlabel('Sampling Number')
        plt.ylabel('Action potential uV')
        target_sig = target_sig[~np.isnan(target_sig)]
        self.max_data = np.max(target_sig)
        self.min_data = np.min(target_sig)
        plt.show()
        return target_sig

    def calc_peaks(self, crop_figure):
        """
        Calculate the positive and negative peaks in the signal.
        """
        partial_peaks, partial_properties = find_peaks(
            crop_figure, prominence=self.prominence, height=self.baseline)
        partial_negative_peaks, partial_negative_properties = find_peaks(
            -crop_figure, prominence=10, height=self.neg_baseline)

        self.fre_pos_sig.append(partial_peaks.shape)
        self.fre_neg_sig.append(partial_negative_peaks.shape)

        self.peaks_data.extend(partial_peaks)
        self.neg_peaks_data.extend(partial_negative_peaks)
        self.p_properties = partial_properties
        self.n_properties = partial_negative_properties

        return self.peaks_data, self.neg_peaks_data, self.p_properties, self.n_properties

    def calc_width(self, crop_figure, pos_p, neg_p):
        """
        Calculate the width of peaks in the signal.
        """
        width_half = peak_widths(
            crop_figure, pos_p, rel_height=self.rel_height)
        negative_width_half = peak_widths(-crop_figure,
                                          neg_p, rel_height=self.rel_height)

        int_width_2 = width_half[2].astype(int)
        int_width_3 = width_half[3].astype(int)
        int_negative_width_2 = negative_width_half[2].astype(int)
        int_negative_width_3 = negative_width_half[3].astype(int)

        for i in range(len(int_width_2)):
            x_data = np.linspace(
                int_width_2[i], int_width_3[i], int_width_3[i] - int_width_2[i])
            self.x_axis_data.append(x_data.astype(int))

        for i in range(len(int_negative_width_2)):
            x_data = np.linspace(
                int_negative_width_2[i], int_negative_width_3[i], int_negative_width_3[i] - int_negative_width_2[i])
            self.negative_x_axis_data.append(x_data.astype(int))
        self.p_width = width_half
        self.n_width = negative_width_half

        return self.p_width, self.n_width, self.x_axis_data, self.negative_x_axis_data

    def calc_width_distribution(self, crop_figure, col, global_label):
        """
        Calculate and plot the distribution of peak widths.
        """
        plt.close()
        total_width = np.concatenate(
            (self.p_width[0] / 10, self.n_width[0] / 10), axis=None)
        bins = np.arange(np.floor(total_width.min()),
                         np.ceil(total_width.max()), 1)
        (mu_w, sigma_w) = norm.fit(total_width)

        plt.hist(total_width, bins, density=1, facecolor='green', alpha=0.75)
        y = norm.pdf(bins, mu_w, sigma_w)
        plt.plot(bins, y, 'r--', linewidth=2, color="black",
                 alpha=0.8, label=global_label)
        plt.xlabel('Width (s)')
        plt.ylabel('Probability')
        plt.title(r'$\mu=%.3f,\ \sigma=%.3f$' % (mu_w, sigma_w))
        plt.legend()
        plt.savefig(
            f'output/statistics_analysis/distribution_width/{col}.png', transparent=True)

        fig = plt.figure(figsize=(12, 9), dpi=300)
        plt.hist(total_width, bins, density=1, facecolor='green', alpha=0.75)
        plt.plot(bins, y, 'r--', linewidth=2, color="black",
                 alpha=0.8, label=global_label)
        plt.xlabel('Width (s)', fontsize=28, fontweight='bold')
        plt.ylabel('Probability', fontsize=28, fontweight='bold')
        plt.title(r'$\mu=%.3f,\ \sigma=%.3f$' %
                  (mu_w, sigma_w), fontsize=26, fontweight='bold')
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend(prop={'size': 22})
        fig.savefig(
            f'output/statistics_analysis/distribution_width/{col}_big.png', transparent=True)
        plt.close()

    def calc_height_distribution(self, crop_figure, col, global_label):
        """
        Calculate and plot the distribution of peak heights.
        """
        plt.close()
        total_v1 = np.concatenate(
            (self.p_properties["prominences"], self.n_properties["prominences"]), axis=None)
        print("Standard Deviation:", statistics.stdev(total_v1))
        print("Mean:", statistics.mean(total_v1))

        bins = np.arange(np.floor(total_v1.min()), np.ceil(total_v1.max()), 1)
        (mu, sigma) = norm.fit(total_v1)

        plt.hist(total_v1, bins, density=1, facecolor='green', alpha=0.75)
        y = norm.pdf(bins, mu, sigma)
        plt.plot(bins, y, 'r--', linewidth=2, color="black",
                 alpha=0.8, label=global_label)
        plt.xlabel('Height (\u03BCV)')
        plt.ylabel('Probability')
        plt.title(r'$\mu=%.3f,\ \sigma=%.3f$' % (mu, sigma))
        plt.legend()
        plt.savefig(
            f'output/statistics_analysis/distribution_height/{col}.png', transparent=True)

        fig = plt.figure(figsize=(12, 9), dpi=300)
        plt.hist(total_v1, bins, density=1, facecolor='green', alpha=0.75)
        plt.plot(bins, y, 'r--', linewidth=2, color="black",
                 alpha=0.5, label=global_label)
        plt.xlabel('Height (\u03BCV)', fontsize=28, fontweight='bold')
        plt.ylabel('Probability', fontsize=28, fontweight='bold')
        plt.title(r'$\mu=%.3f,\ \sigma=%.3f$' %
                  (mu, sigma), fontsize=28, fontweight='bold')
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend(prop={'size': 22})
        fig.savefig(
            f'output/statistics_analysis/distribution_height/{col}_big.png', transparent=True)
        plt.close()

    def draw_height_time(self, crop_figure, col, global_label):
        """
        Plot peak heights over time.
        """
        plt.close()
        total_p = np.concatenate(
            (self.p_properties["prominences"], self.n_properties["prominences"]))
        total_count_p = np.concatenate((self.peaks_data, self.neg_peaks_data))

        plt.plot(np.array(total_count_p) / 864000, total_p,
                 color="black", alpha=0.8, label=global_label)
        plt.ylabel('Height (\u03BCV)')
        plt.xlabel('Time (day)')
        plt.legend()
        plt.savefig(
            f'output/statistics_analysis/t_height/{col}.png', transparent=False)

        fig = plt.figure(figsize=(12, 9), dpi=300)
        plt.plot(np.array(total_count_p) / 864000, total_p,
                 color="black", alpha=0.8, label=global_label)
        plt.xlabel('Time (day)', fontsize=28, fontweight='bold')
        plt.ylabel('Height (\u03BCV)', fontsize=28, fontweight='bold')
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend(prop={'size': 20})
        fig.savefig(
            f'output/statistics_analysis/t_height/{col}_big.png', transparent=True)
        plt.close()

    def draw_width_time(self, crop_figure, col, global_label):
        """
        Plot peak widths over time.
        """
        plt.close()
        total_w = np.concatenate((self.p_width[0], self.n_width[0]))
        total_count_w = np.concatenate((self.p_width[2], self.n_width[2]))

        plt.plot(np.array(total_count_w) / 864000, total_w / 10,
                 color="black", alpha=0.8, label=global_label)
        plt.ylabel('Width (s)')
        plt.xlabel('Time (day)')
        plt.legend()
        plt.savefig(f'output/statistics_analysis/t_width/{col}.png')

        fig = plt.figure(figsize=(12, 9), dpi=300)
        plt.plot(np.array(total_count_w) / 864000, total_w / 10,
                 color="black", alpha=0.8, label=global_label)
        plt.xlabel('Time (day)', fontsize=28, fontweight='bold')
        plt.ylabel('Width (s)', fontsize=28, fontweight='bold')
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend(prop={'size': 22}, loc='upper right')
        fig.savefig(
            f'output/statistics_analysis/t_width/{col}_big.png', transparent=True)
        plt.close()

    def draw_height_width(self, crop_figure, col, global_label):
        """
        Plot peak heights against widths.
        """
        plt.close()
        total_w = np.concatenate((self.p_width[0], self.n_width[0]))
        total_p = np.concatenate(
            (self.p_properties["prominences"], self.n_properties["prominences"]))

        plt.scatter(total_w / 10, total_p, color="black",
                    alpha=0.8, label=global_label)
        plt.ylabel('Height (\u03BCV)')
        plt.xlabel('Width (s)')
        plt.legend()
        plt.savefig(
            f'output/statistics_analysis/height_width/{col}.png', transparent=True)

        fig = plt.figure(figsize=(12, 9), dpi=300)
        plt.scatter(total_w / 10, total_p, color="black",
                    alpha=0.8, label=global_label)
        plt.xlabel('Width (s)', fontsize=28, fontweight='bold')
        plt.ylabel('Height (\u03BCV)', fontsize=28, fontweight='bold')
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend(prop={'size': 22})
        fig.savefig(
            f'output/statistics_analysis/height_width/{col}_big.png', transparent=True)
        plt.close()

    def calc_frequency(self, data, pos_p, neg_p):
        """
        Calculate the frequency of peaks in the signal.
        """
        total_spike = len(pos_p) + len(neg_p)
        periodic_freq = total_spike / self.time_freq  # per second
        return periodic_freq


def main():
    sig_pp = SignalPreProcessing()
    total_file_data = []
    col = 1  # CSV file column data
    global_label = "Fungi Signal"

    # Load and concatenate data from the specified file
    file_path = 'data/new_data.csv'  # data input
    signal_data = sig_pp.load_data(file_path, col)
    total_file_data = np.concatenate((total_file_data, signal_data), axis=0)

    print("Total data:", np.array(total_file_data))
    print("Total length:", total_file_data.shape)

    x_axis = np.arange(len(total_file_data))

    plt.close()
    plt.plot(x_axis / 864000, total_file_data,
             color="black", alpha=0.8, label=global_label)
    plt.xlabel('Time (day)', fontweight='bold')
    plt.ylabel('Action Potentials (\u03BCV)', fontweight='bold')
    plt.legend()
    plt.savefig(
        f'output/statistics_analysis/macro_v_time/t_mv_{col}.png', transparent=True)

    plt.close()
    fig = plt.figure(figsize=(14, 10.5), dpi=300)
    plt.plot(x_axis / 864000, total_file_data,
             color="black", alpha=0.8, label=global_label)
    plt.xlabel('Time (day)', fontsize=28, fontweight='bold')
    plt.ylabel('Action Potentials (\u03BCV)', fontsize=28, fontweight='bold')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(prop={'size': 24})
    fig.savefig(
        f'output/statistics_analysis/macro_v_time/t_mv_{col}_big.png', transparent=True)
    plt.close()

    pos_p, neg_p, pos_prop, neg_prop = sig_pp.calc_peaks(total_file_data)
    pos_w, neg_w, pos_x_w, neg_x_w = sig_pp.calc_width(
        total_file_data, pos_p, neg_p)

    sig_pp.calc_width_distribution(total_file_data, col, global_label)
    sig_pp.calc_height_distribution(total_file_data, col, global_label)
    sig_pp.draw_height_time(total_file_data, col, global_label)
    sig_pp.draw_width_time(total_file_data, col, global_label)
    sig_pp.draw_height_width(total_file_data, col, global_label)

    for x in range(int(len(total_file_data) / sig_pp.crop_interval)):
        crop_figure = total_file_data[sig_pp.crop_cnt:
                                      sig_pp.crop_cnt + sig_pp.crop_interval]
        pos_p, neg_p, pos_prop, neg_prop = sig_pp.calc_peaks(crop_figure)
        freq = sig_pp.calc_frequency(crop_figure, pos_p, neg_p)
        sig_pp.total_freq.append(freq)
        sig_pp.crop_cnt += sig_pp.crop_interval

    plt.close()
    total_freq = np.hstack(sig_pp.total_freq)
    plt.text(230, 0.04, f'mean = {np.mean(total_freq):.3f} , std = {np.std(total_freq):.3f}',
             fontsize=12, horizontalalignment='center', verticalalignment='center')
    plt.plot(total_freq, color="black", alpha=0.8, label=global_label)
    plt.xlabel('Sampling time (min)')
    plt.ylabel('Number of peaks (peaks/s)')
    plt.legend()
    plt.savefig(
        f'output/statistics_analysis/peak_per_min/col_{col}.png', transparent=True)
    plt.close()

    fig = plt.figure(figsize=(12, 9), dpi=300)
    plt.text(170, 0.035, f'mean = {np.mean(total_freq):.3f} , std = {np.std(total_freq):.3f}',
             fontsize=22, horizontalalignment='center', verticalalignment='center')
    plt.plot(total_freq, color="black", alpha=0.8, label=global_label)
    plt.xlabel('Sampling Time (min)', fontsize=28, fontweight='bold')
    plt.ylabel('Number of peaks (peaks/s)', fontsize=28, fontweight='bold')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(prop={'size': 24}, loc="upper right")
    fig.savefig(
        f'output/statistics_analysis/peak_per_min/col_{col}_big.png', transparent=True)
    plt.close()


if __name__ == "__main__":
    main()
