import pandas as pd


def load_data_from_file(file_path: str) -> pd.DataFrame:
    """
    Load the CSV file from a file and return a pandas DataFrame

    Parameters:
        file_path (str): The path to the CSV data file

    Returns:
        pd.DataFrame: DataFrame containing the data
    """

    # Load data from CSV
    df = pd.read_csv(file_path)

    # Rename the first column to timestamp
    df = df.rename(columns={df.columns[0]: "timestamp"})

    # Convert the timestamp column to numeric
    # df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    return df


def save_data_to_pickle(df: pd.DataFrame, output_file_path: str) -> None:
    df.to_pickle(output_file_path)


def load_data_from_pickle(file_path: str) -> pd.DataFrame:
    df = pd.read_pickle(file_path)

    return df


# Example usage
if __name__ == "__main__":
    file_path = "data/SEMROOM_UnshieldedTwistedPair.csv"
    data = load_data_from_file(file_path)

    print(data["timestamp"].min())
