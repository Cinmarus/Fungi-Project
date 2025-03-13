import pandas as pd
import matplotlib.pyplot as plt


def visualize_data(df: pd.DataFrame) -> None:
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain a 'timestamp' column")

    time = df["timestamp"]

    for col in df.columns:
        if col != "timestamp":
            plt.plot(time, df[col], label=col)

    plt.legend()

    plt.show()


def visualize_data_baseline(df: pd.DataFrame, column: str, timestamp_column: str = "timestamp") -> None:
    fig, ax = plt.subplots()

    # raw_line, = ax.plot(
    #     df[timestamp_column],
    #     df[column],
    #     label="Raw Data",
    #     linewidth=2,
    # )

    # baseline_color = raw_line.get_color()
    baseline_col = f"{column}_baseline"

    ax.plot(
        df[timestamp_column],
        df[baseline_col],
        label="Baseline",
        linestyle="--",
        linewidth=2,
        # color=baseline_color,
    )

    ax.legend()
    ax.grid(True)

    plt.show()
