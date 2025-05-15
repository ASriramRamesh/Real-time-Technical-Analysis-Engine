import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import time
from contextlib import contextmanager
import pickle


@contextmanager
def timer(name):
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    print(f"{name}: {end_time - start_time:.4f} seconds")


def get_stocks_by_sector(df_stocks, df_stocks_sector, sector_id):
    """
    Returns a DataFrame with stock IDs and symbols for a given sector ID.

    Parameters:
    df_stocks (pd.DataFrame): DataFrame containing stock information.
    df_stocks_sector (pd.DataFrame): DataFrame containing sector information.
    sector_id (int): ID of the sector.

    Returns:
    pd.DataFrame: DataFrame with stock IDs and symbols for the given sector.
    """
    # Filter sector DataFrame by sector ID
    sector_stocks = df_stocks_sector[df_stocks_sector["sector_id"] == sector_id]

    # Merge sector DataFrame with stock DataFrame on stock ID
    df_result = pd.merge(sector_stocks, df_stocks, on="stock_id")

    # Select only stock ID and symbol columns
    df_result = df_result[["stock_id", "symbol"]]
    return df_result


def load_pickle_from_file(filename, folder_path="."):
    """
    Load a dictionary from a pickle file.

    Args:
    - filename (str): Name of the file.
    - folder_path (str, optional): Folder path where the file is located. Defaults to current directory.

    Returns:
    - dict: Loaded dictionary.
    """
    file_path = os.path.join(folder_path, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")

    with open(file_path, "rb") as file:
        pickle_object = pickle.load(file)

    return pickle_object


def load_csv_to_dataframe(folder_path, file_name, is_date_index=False):
    # Construct the full file path
    # file_path = os.path.join(folder_path, file_name)
    file_path = folder_path + file_name

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    if is_date_index:
        # Convert the 'datetime' column to datetime type
        df["datetime"] = pd.to_datetime(df["datetime"])

        # Set the 'datetime' column as the index
        df.set_index("datetime", inplace=True)

    return df


def get_chart_data_response(start_datetime_naive, time_frame, indicator_id, sector_id):
    folder_path = "pickle_files/"
    file_name = "stock_master.csv"

    df_stocks = load_csv_to_dataframe(folder_path, file_name)

    file_name = "stock_sector.csv"
    df_stocks_sector = load_csv_to_dataframe(folder_path, file_name)

    df_result = get_stocks_by_sector(df_stocks, df_stocks_sector, sector_id)
    stock_ids = [x for x in df_result["stock_id"]]

    year = start_datetime_naive.year
    month = start_datetime_naive.month
    day = start_datetime_naive.day

    column_count = 10

    file_name = "indicator_master.csv"
    df_indicators = load_csv_to_dataframe(folder_path, file_name)

    indicator_column_name = df_indicators.loc[
        df_indicators["indicator_id"] == indicator_id, "indicator_column_name"
    ].values[0]

    current_dir = os.getcwd()
    pickle_files_dir = os.path.join(current_dir, "pickle_files")

    folder_path_pickle = os.path.join(pickle_files_dir, str(year), str(month), str(day))
    folder_path_timeframe = os.path.join(folder_path_pickle, str(time_frame))

    file_name = "cid_ta.pkl"
    combined_df_timeframe = load_pickle_from_file(
        file_name, folder_path=folder_path_timeframe
    )

    start_datetime_aware = start_datetime_naive.tz_localize(
        combined_df_timeframe.index.tz
    )
    filtered_df = combined_df_timeframe.loc[start_datetime_aware:]

    first_10_records = filtered_df.head(column_count)
    data = {}

    for datetime_index, df in first_10_records.iterrows():
        # Extract the time from the datetime index
        time = datetime_index.strftime("%H:%M")

        # Add values to the data dictionary
        data[time] = [
            df["dataframe"].loc[stock_id, indicator_column_name]
            for stock_id in stock_ids
        ]

    # Convert the data dictionary to a DataFrame and set index
    df_time_indicators = pd.DataFrame(data).set_index(
        pd.Index(stock_ids, name="stock_id")
    )
    df_time_indicators.reset_index(inplace=True)

    df_final = pd.merge(df_result, df_time_indicators, on="stock_id")
    df_final.drop("stock_id", axis=1, inplace=True)

    data = df_final.to_dict(orient="records")
    for row in data:
        for col in df_final.columns[1:]:
            row[col] = float(row[col])

    return data


def main():

    # We would be getting this variables from the request and we query the pickle file and return 10 columns for the selected sector.
    start_datetime_naive = pd.to_datetime("2024-10-25 10:10:00")
    time_frame = 15
    indicator_id = 12
    sector_id = 12

    with timer("get_chart_data_response"):
        data = get_chart_data_response(
            start_datetime_naive, time_frame, indicator_id, sector_id
        )

    print(data)


if __name__ == "__main__":
    main()
