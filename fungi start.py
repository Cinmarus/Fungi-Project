from analysis import extract_signal_from_data
from data_loader import load_data_from_file
from visualization import visualize_data, visualize_data_baseline

# assumed that the baseline fluctuations have frequencies lower than this
BASELINE_CUTOFF_FREQ = 20  # [Hz]

file_path = "data/SEMROOM_UnshieldedTwistedPair.csv"
data = load_data_from_file(file_path)

analysed_data = extract_signal_from_data(data, BASELINE_CUTOFF_FREQ)
analysis_col = data.columns[-1]

visualize_data_baseline(analysed_data, analysis_col)
