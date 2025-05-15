import pandas as pd
import numpy as np

import pytz
import sqlite3
import pickle
import os
import sys


# Function to load the dictionary from a file
def load_dict_from_file(filename):
    with open(filename, "rb") as file:
        loaded_dict = pickle.load(file)

    return loaded_dict


def load_csv_to_dataframe(folder_path, file_name, is_date_index=False):
    file_path = folder_path + file_name

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    if is_date_index:
        # Convert the 'datetime' column to datetime type
        df["datetime"] = pd.to_datetime(df["datetime"])

        # Set the 'datetime' column as the index
        df.set_index("datetime", inplace=True)

    return df


# load market data from the pickle file and load ohlc values for 15, 5, 3 min time frames from folders ohlc_csv
def retrieve_market_data():
    market_data = load_dict_from_file("market_data.pkl")

    for stock_id, stock in market_data.items():

        # We do not have intraday data for this stock hence need to comment
        if stock["symbol"] == "VBL":
            continue

        folder_path_15min = "ohlc_csv/15/"
        folder_path_5min = "ohlc_csv/5/"
        folder_path_3min = "ohlc_csv/3/"
        folder_high_levels = "levels_csv/highs/"
        folder_low_levels = "levels_csv/lows/"

        file_name = f"{stock_id}.csv"

        market_data[stock_id]["highs"] = load_csv_to_dataframe(
            folder_high_levels, file_name
        )
        market_data[stock_id]["lows"] = load_csv_to_dataframe(
            folder_low_levels, file_name
        )
        market_data[stock_id][15] = load_csv_to_dataframe(
            folder_path_15min, file_name, is_date_index=True
        )
        market_data[stock_id][5] = load_csv_to_dataframe(
            folder_path_5min, file_name, is_date_index=True
        )
        market_data[stock_id][3] = load_csv_to_dataframe(
            folder_path_3min, file_name, is_date_index=True
        )

    return market_data


def main():

    market_data = retrieve_market_data()
    print(market_data[1][3]["ohlc"].head())


if __name__ == "__main__":
    main()
