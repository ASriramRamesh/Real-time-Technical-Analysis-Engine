import asyncio
from typing import Dict
import pandas as pd
import numpy as np
from pymongo import MongoClient
import os
from datetime import datetime, timedelta
import time
from contextlib import contextmanager
import pickle

from ta.event_indicators import Vector_Indicators, BaseIndicator, SwingLevel

from QueryIndicators import (
    connect_to_mongodb,
    get_trading_days_data,
    resample_stock_data,
    save_to_csv,
    load_csv_to_dataframe,
    get_stock_master_data,
)
from SaveStreaming import load_dict_from_file

from BuildTradingDaysFolders import (
    get_start_streaming_trading_days,
    get_stock_data_till,
)

# from BuildPickleFiles import call_streaming_vectorized_indicators


swing_level_indicator_id = 7


@contextmanager
def timer(name):
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    print(f"{name}: {end_time - start_time:.4f} seconds")


def print_df_info_head_tail(df, message):
    print(message)
    print("Dataframe info:")
    print(df.info())
    print("Dataframe description:")
    print(df.describe())
    print("Dataframe head:")
    print(df.head())
    print("Dataframe tail:")
    print(df.tail())


# Function to parse datetime with timezone
def parse_datetime(dt_str):
    # Parse the datetime string
    dt = pd.to_datetime(dt_str, format="%Y-%m-%d %H:%M:%S%z")
    # Localize to Asia/Kolkata timezone
    return dt.tz_convert("Asia/Kolkata")


# This is where we append stock OHLC based on time and then map it to a datetime.
def get_combined_df(reference_df, stock_dfs):
    # Prepare data for the combined dataframe
    combined_data = {}

    # Iterate through each unique datetime
    for dt in reference_df.index:
        # Create a new dataframe for this datetime
        df_data = []
        for stock_id, stock_df in stock_dfs.items():
            # This is the most important line of code. It handles missing values
            # It is possible that some stocks may not have trades for a time interval.
            if dt in stock_df.index:
                data = {
                    "stock_id": stock_id,
                    "open": stock_df.loc[dt, "open"],
                    "high": stock_df.loc[dt, "high"],
                    "low": stock_df.loc[dt, "low"],
                    "close": stock_df.loc[dt, "close"],
                }
                df_data.append(data)

        if df_data:  # Only create a dataframe if there's data for this datetime
            df = pd.DataFrame(df_data)
            combined_data[dt] = df

    # Create the final combined dataframe
    combined_df = pd.DataFrame(
        {"dataframe": combined_data},
        index=pd.Index(combined_data.keys(), name="datetime"),
    )

    return combined_df


# This function flips multiple dataframe of stock OHLC price to a dataframe of dataframe and pickles it.
# This is when we loop through 10 days for 15min, 6 days for 5 min, 3 days for 3 min.
# The reason is that we do not have the prior days indicator_data as a pickle file.
def build_save_combined_df(df_stock_master, trading_date, folder_path=None):

    stock_dfs_15 = {}
    stock_dfs_5 = {}
    stock_dfs_3 = {}
    stock_dfs_1 = {}
    year = trading_date.year
    month = trading_date.month
    day = trading_date.day

    # Load all stock dataframes
    for _, row in df_stock_master.iterrows():
        stock_id = row["stock_id"]
        stock_symbol = row["stock_symbol"]
        # Later on we do not have intraday data for this stock
        if stock_symbol == "VBL" or stock_symbol == "IDFC" or stock_symbol == "DRREDDY":
            continue

        file_name = f"{stock_id}.csv"
        folder_path_15min = f"historical/{year}/{month}/{day}/15/{stock_id}/"
        folder_path_5min = f"historical/{year}/{month}/{day}/5/{stock_id}/"
        folder_path_3min = f"historical/{year}/{month}/{day}/3/{stock_id}/"
        folder_path_1min = f"historical/{year}/{month}/{day}/1/{stock_id}/"

        # if os.path.exists(file_name):
        df_15 = load_csv_to_dataframe(folder_path_15min, file_name)
        df_5 = load_csv_to_dataframe(folder_path_5min, file_name)
        df_3 = load_csv_to_dataframe(folder_path_3min, file_name)
        df_1 = load_csv_to_dataframe(folder_path_1min, file_name)

        if df_15 is None:
            continue

        df_15["datetime"] = df_15["datetime"].apply(parse_datetime)
        # Convert datetime to string for consistent indexing
        # df_15["datetime_str"] = df_15["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
        df_15.set_index("datetime", inplace=True)
        stock_dfs_15[stock_id] = df_15

        df_5["datetime"] = df_5["datetime"].apply(parse_datetime)
        # df_5["datetime_str"] = df_5["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
        df_5.set_index("datetime", inplace=True)
        stock_dfs_5[stock_id] = df_5

        df_3["datetime"] = df_3["datetime"].apply(parse_datetime)
        # df_3["datetime_str"] = df_3["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
        df_3.set_index("datetime", inplace=True)
        stock_dfs_3[stock_id] = df_3

        df_1["datetime"] = df_1["datetime"].apply(parse_datetime)
        # df_3["datetime_str"] = df_3["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
        df_1.set_index("datetime", inplace=True)
        stock_dfs_1[stock_id] = df_1

    # Assuming all dataframes have the same datetime range, we can use any of them as reference
    reference_df_15 = next(iter(stock_dfs_15.values()))
    reference_df_5 = next(iter(stock_dfs_5.values()))
    reference_df_3 = next(iter(stock_dfs_3.values()))
    reference_df_1 = next(iter(stock_dfs_1.values()))

    # Prepare data for the combined dataframe
    combined_data_15 = get_combined_df(reference_df_15, stock_dfs_15)
    combined_data_5 = get_combined_df(reference_df_5, stock_dfs_5)
    combined_data_3 = get_combined_df(reference_df_3, stock_dfs_3)
    combined_data_1 = get_combined_df(reference_df_1, stock_dfs_1)
    # return

    file_name = "ci_ohlc_pivot.pkl"  # The file name does not change
    # The folder path is different
    folder_path_15min = os.path.join(folder_path, "15")
    folder_path_5min = os.path.join(folder_path, "5")
    folder_path_3min = os.path.join(folder_path, "3")
    folder_path_1min = os.path.join(folder_path, "1")
    save_pickle_to_file(combined_data_15, file_name, folder_path=folder_path_15min)
    save_pickle_to_file(combined_data_5, file_name, folder_path=folder_path_5min)
    save_pickle_to_file(combined_data_3, file_name, folder_path=folder_path_3min)
    save_pickle_to_file(combined_data_1, file_name, folder_path=folder_path_1min)


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


# This is saved to the year, month, day and time frame folder
def save_pickle_to_file(pickle_file, file_name, folder_path="."):
    """
    Save a dictionary to a pickle file.

    Args:
    - dictionary (dict): Dictionary to be saved.
    - filename (str): Name of the file.
    - folder_path (str, optional): Folder path where the file will be saved. Defaults to current directory.
    """
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "wb") as file:
        pickle.dump(pickle_file, file)


# This method is for saving in the pickle files folder.
def save_combined_df_vector_indicators_to_pickle(
    combined_df_15,
    combined_df_5,
    combined_df_3,
    vector_indicators_15,
    vector_indicators_5,
    vector_indicators_3,
):
    # We have to remove all datetime which is less than start of streaming date.
    last_trading_day = combined_df_15.index[-1].date()
    year = last_trading_day.year
    month = last_trading_day.month
    day = last_trading_day.day

    # The path till the folder
    folder_path_day = os.path.join(
        os.getcwd(), "pickle_files", str(year), str(month), str(day)
    )

    folder_path_15min = os.path.join(folder_path_day, "15")
    folder_path_5min = os.path.join(folder_path_day, "5")
    folder_path_3min = os.path.join(folder_path_day, "3")

    # Filter for records for the last trading day.
    # Save the pickle of these files in folder
    file_name = "cid_ta.pkl"
    save_pickle_to_file(combined_df_15, file_name, folder_path=folder_path_15min)
    save_pickle_to_file(combined_df_5, file_name, folder_path=folder_path_5min)
    save_pickle_to_file(combined_df_3, file_name, folder_path=folder_path_3min)

    # This would have the serialized streaming indicators classes
    file_name = "vi.pkl"
    save_pickle_to_file(vector_indicators_15, file_name, folder_path=folder_path_15min)
    save_pickle_to_file(vector_indicators_5, file_name, folder_path=folder_path_5min)
    save_pickle_to_file(vector_indicators_3, file_name, folder_path=folder_path_3min)
    print(last_trading_day)


# Generate combined
async def process_vector_indicators(
    vector_indicators: Vector_Indicators,
    pickle_path: str,
    folder_path=None,
    folder_path_1min=None,
    time_interval: int = None,
):
    # We need the mapping sector
    folder_path_stocks = os.path.join(os.getcwd(), "csv/")
    file_name_stocks = "stock_master.csv"

    df_stock_master = load_csv_to_dataframe(
        folder_path_stocks, file_name=file_name_stocks
    )
    df_stock_master.set_index("stock_id", inplace=True)
    df_stock_master = df_stock_master[["sector_id"]]

    if folder_path is None:
        combined_df = load_dict_from_file(pickle_path)
    else:
        file_name = "ci_ohlc_pivot.pkl"
        combined_df = load_pickle_from_file(file_name, folder_path=folder_path)
        combined_df_1min = load_pickle_from_file(
            file_name, folder_path=folder_path_1min
        )
    for datetime_index, df in combined_df.iterrows():
        # We have to pass the linear regression values for 1 min for all time frames
        for i in range(time_interval):
            one_min_index = datetime_index + pd.Timedelta(minutes=i)
            if one_min_index in combined_df_1min.index:
                vector_indicators.update_stats(
                    combined_df_1min.loc[one_min_index]["dataframe"].set_index(
                        "stock_id"
                    ),
                    df_stock_master,
                )
        # This would add the streaming indicators columns
        await vector_indicators.update(
            df["dataframe"].set_index("stock_id"), datetime_index
        )
        combined_df.at[datetime_index, "dataframe"] = vector_indicators.df

    return combined_df


# This logic could be used to get a time slice of selected stocks for comparing price action
def revert_to_stock_df(combined_df, trading_date, time_frame):
    # Assuming 'df' is your nested DataFrame
    stock_dfs = {}

    columns = [
        "open",
        "high",
        "low",
        "close",
        "hlc3",
        "st_fpvalue",
        "st_fpdirection",
        "st_spvalue",
        "st_spdirection",
        "ema_9",
        "ema_21",
        "ema_50",
        "ppo_hlc3ema9",
        "ppo_hlc3ema21",
        "ppo_hlc3st_fp",
        "ppo_hlc3st_sp",
        "ind_slope",
        "ind_y-intercept",
        "ind_r-squared",
        "ind_p-coef",
        "dep_slope",
        "dep_y-intercept",
        "dep_r-squared",
        "dep_p-coef",
    ]

    empty_stock_ids = set([23, 64, 82])
    # Iterate over each stock_id
    for stock_id in range(1, 97):
        # Create a DataFrame for the stock_id
        stock_df = create_empty_df()
        if stock_id in empty_stock_ids:
            continue

        # Iterate over each row in the nested DataFrame
        for index, row in combined_df.iterrows():
            inner_df = row["dataframe"]
            inner_df = inner_df.reset_index()
            # Filter the inner DataFrame by stock_id
            stock_data = inner_df.loc[inner_df["stock_id"] == stock_id, columns]

            if stock_data.empty:
                continue

            # Append the data to the DataFrame
            stock_df = pd.concat(
                [
                    stock_df if not stock_df.empty else None,
                    pd.DataFrame(
                        [[index] + stock_data.values[0].tolist()],
                        columns=["datetime"] + columns,
                    ),
                ],
                ignore_index=True,
            )
        # For some stock_id we may not have values
        if stock_df.empty:
            empty_stock_ids.add(stock_id)
            continue
        # Set the datetime as the index (optional)
        stock_df.set_index("datetime", inplace=True)

        # Store the DataFrame in a dictionary
        stock_dfs[stock_id] = stock_df

    # Get the year, month, and day to save dataframe as csv to correct folder.
    year = trading_date.year
    month = trading_date.month
    day = trading_date.day

    # Now you have 96 DataFrames, one for each stock_id have to save it in indicators_stream.csv
    for stock_id in range(1, 97):
        if stock_id in empty_stock_ids:
            continue

        file_name = "indicators_linear_reg_new.csv"
        folder_path = f"historical/{year}/{month}/{day}/{time_frame}/{stock_id}/"
        df_csv = stock_dfs[stock_id]
        save_to_csv(df_csv, file_name, folder_path)


# The logic in this method could be used to ascertain the intraday swing levels. For select stock, timeframe.
def save_vector_data_indicators_csv(
    vector_indicators: Vector_Indicators,
    trading_date,
    time_frame,
):
    # Get the year, month, and day to save dataframe as csv to correct folder.
    year = trading_date.year
    month = trading_date.month
    day = trading_date.day
    empty_stock_ids = set([23, 64, 82])

    for stock_id in range(1, 97):
        if stock_id in empty_stock_ids:
            continue
        swing_level: SwingLevel = vector_indicators.indicator_data[stock_id][
            time_frame
        ][swing_level_indicator_id]

        if swing_level is not None:
            df_swing_levels = swing_level.df_swing_levels
            # df = filter_close_values(df_swing_levels)
            file_name = "swing_levels.csv"
            folder_path = f"historical/{year}/{month}/{day}/{time_frame}/{stock_id}/"
            save_to_csv(df_swing_levels, file_name, folder_path)


def filter_close_values(df, time_threshold_minutes=30, value_threshold=20):
    """
    Filter out rows where either timestamps are too close or values are too similar,
    keeping the newer values.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'trend_start_time' and 'start_trend_value' columns
    time_threshold_minutes : int
        Minimum time difference in minutes between entries
    value_threshold : float
        Maximum allowed difference between consecutive start_trend_values

    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame with close timestamps and similar values removed
        datetime,is_up_trend,is_up_move,level,trigger_datetime
    """
    # Ensure trend_start_time is datetime
    df.index = pd.to_datetime(df.index)
    df["trigger_datetime"] = pd.to_datetime(df["trigger_datetime"])

    # Sort by datetime to ensure chronological order
    # df = df.sort_values("datetime")

    # Calculate time difference with next row
    df["time_diff"] = df.index.to_series().shift(-1) - df.index

    # Calculate absolute value difference with next row
    df["value_diff"] = abs(df["level"].shift(-1) - df["level"])

    # Create mask for rows to keep
    # Keep row if:
    # 1. It's the last row (time_diff will be NaT), or
    # 2. Time difference is greater than threshold AND value difference is greater than threshold
    mask = df["time_diff"].isna() | (
        (df["time_diff"] > timedelta(minutes=time_threshold_minutes))
        & (df["value_diff"] > value_threshold)
    )

    # Return filtered DataFrame without the temporary columns
    return df[mask].drop(["time_diff", "value_diff"], axis=1)


def create_empty_df():
    return pd.DataFrame(
        columns=[
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "hlc3",
            "st_fpvalue",
            "st_fpdirection",
            "st_spvalue",
            "st_spdirection",
            "ema_9",
            "ema_21",
            "ema_50" "ppo_hlc3ema21",
            "ppo_hlc3st_fp",
            "ppo_hlc3st_sp",
            "ind_slope",
            "ind_y-intercept",
            "ind_r-squared",
            "ind_p-coef",
            "dep_slope",
            "dep_y-intercept",
            "dep_r-squared",
            "dep_p-coef",
            "ppo_hlc3ema9",
        ],
    )


# Adding a new column datetime which would be the index.
async def reset_swing_levels(current_trading_day, previous_trading_day=None):
    current_dir = os.getcwd()
    pickle_files_dir = os.path.join(current_dir, "pickle_files")

    year = previous_trading_day.year
    month = previous_trading_day.month
    day = previous_trading_day.day

    folder_path_pickle = os.path.join(pickle_files_dir, str(year), str(month), str(day))
    folder_path_15min = os.path.join(folder_path_pickle, str(15))
    folder_path_5min = os.path.join(folder_path_pickle, str(5))
    folder_path_3min = os.path.join(folder_path_pickle, str(3))

    # The previous days vector indicators to reset the current days vector indicators.
    file_name = "vi.pkl"
    vi_15_pd: Vector_Indicators = load_pickle_from_file(
        file_name, folder_path=folder_path_15min
    )
    vi_5_pd: Vector_Indicators = load_pickle_from_file(
        file_name, folder_path=folder_path_5min
    )
    vi_3_pd: Vector_Indicators = load_pickle_from_file(
        file_name, folder_path=folder_path_3min
    )

    # From here we need the current days values for combined df and vector indicators.
    year = current_trading_day.year
    month = current_trading_day.month
    day = current_trading_day.day

    folder_path_pickle = os.path.join(pickle_files_dir, str(year), str(month), str(day))
    folder_path_15min = os.path.join(folder_path_pickle, str(15))
    folder_path_5min = os.path.join(folder_path_pickle, str(5))
    folder_path_3min = os.path.join(folder_path_pickle, str(3))

    # We need the full technical indicators. We are not resetting any technical indicator only new column in swing levels
    file_name = "cid_ta.pkl"
    combined_df_15min = load_pickle_from_file(file_name, folder_path=folder_path_15min)
    combined_df_5min = load_pickle_from_file(file_name, folder_path=folder_path_5min)
    combined_df_3min = load_pickle_from_file(file_name, folder_path=folder_path_3min)

    file_name = "vi.pkl"
    vi_15: Vector_Indicators = load_pickle_from_file(
        file_name, folder_path=folder_path_15min
    )
    vi_5: Vector_Indicators = load_pickle_from_file(
        file_name, folder_path=folder_path_5min
    )
    vi_3: Vector_Indicators = load_pickle_from_file(
        file_name, folder_path=folder_path_3min
    )

    # This is only for 28th March 2024. Post this we have to load previous days swing_level,
    # first_period, start of up move, start of down move
    # We pass the old vector indicator and it loops through stock_id, time frame and indicator id
    # and resets it to previous day values of df_swing_levels, first period, start of up move, start of down move.
    vi_15.reset_swing_levels(vi_15_pd.indicator_data)
    vi_5.reset_swing_levels(vi_5_pd.indicator_data)
    vi_3.reset_swing_levels(vi_3_pd.indicator_data)

    for datetime_index, df in combined_df_15min.iterrows():
        # This will reset the swing levels for the vector indicators in the 15 min time frame
        await vi_15.update_swing_level(df["dataframe"], datetime_index)

    for datetime_index, df in combined_df_5min.iterrows():
        # This will reset the swing levels for the vector indicators in the 15 min time frame
        await vi_5.update_swing_level(df["dataframe"], datetime_index)

    for datetime_index, df in combined_df_3min.iterrows():
        # This will reset the swing levels for the vector indicators in the 15 min time frame
        await vi_3.update_swing_level(df["dataframe"], datetime_index)

    # The internal state of vector indicators swing level has been updated. We need to verify this.
    file_name = "vi.pkl"  # Initially we are not overwriting the file after testing we will overwrite.
    save_pickle_to_file(vi_15, file_name, folder_path=folder_path_15min)
    save_pickle_to_file(vi_5, file_name, folder_path=folder_path_5min)
    save_pickle_to_file(vi_3, file_name, folder_path=folder_path_3min)

    # # The csv file will help us verify that we have fixed the problem and not broken the implementation.
    # save_vector_data_indicators_csv(vi_15, current_trading_day, 15)
    # save_vector_data_indicators_csv(vi_5, current_trading_day, 5)
    # save_vector_data_indicators_csv(vi_3, current_trading_day, 3)
    print(current_trading_day)


# This takes a dataframe of trading days and resets the swing levels for each day.
async def reset_swing_levels_from(
    df_trading_days, start_trading_day, prior_day_to_start_trading_day
):
    df_trading_days["date"] = pd.to_datetime(df_trading_days["date"])
    df = df_trading_days[df_trading_days["date"] > prior_day_to_start_trading_day]

    df = df.sort_values(by="date")
    previous_date = prior_day_to_start_trading_day

    for index, row in df.iterrows():
        current_date = row["date"]

        await reset_swing_levels(current_date, previous_date)
        previous_date = current_date


# This is just to create the 1 min folder in all the trading days folder.
def create_folders(df_trading_days):
    for index, row in df_trading_days.iterrows():
        date = row["date"]
        # if date > start_streaming_trading_day:
        year = date.year
        month = date.month
        day = date.day
        folder_path = f"pickle_files/{year}/{month}/{day}"
        subfolder_path = f"{folder_path}/1"
        os.makedirs(subfolder_path, exist_ok=True)
        # for i in range(1, 100):
        #     subsubfolder_path = f"{subfolder_path}/{i}"
        #     os.makedirs(subsubfolder_path, exist_ok=True)


async def generate_ohlc_daily_data(db, current_trading_day, start_date=None):

    next_date = current_trading_day + timedelta(days=1)
    year = current_trading_day.year
    month = current_trading_day.month
    day = current_trading_day.day

    market_data = load_dict_from_file("market_data.pkl")
    for stock_id, stock in market_data.items():
        stock_symbol = stock["symbol"]

        file_name = f"{stock_id}.csv"
        # folder_path_15min = f"historical/{year}/{month}/{day}/15/{stock_id}/"
        # folder_path_5min = f"historical/{year}/{month}/{day}/5/{stock_id}/"
        folder_path_1min = f"historical/{year}/{month}/{day}/1/{stock_id}/"
        # We do not have intraday data for this stock hence need to comment
        if stock_symbol == "VBL" or stock_symbol == "IDFC" or stock_symbol == "DRREDDY":
            continue
        # Fetches the stock data for that date range.
        df_1min = get_stock_data_till(db, stock_symbol, next_date, start_date)
        # df_15min = resample_stock_data(df_1min)
        # df_5min = resample_stock_data(df_1min, period="5min")
        # df_3min = resample_stock_data(df_1min, period="3min")
        # # Save to appropriate folder
        # save_to_csv(df_15min, file_name, folder_path_15min)
        # save_to_csv(df_5min, file_name, folder_path_5min)
        save_to_csv(df_1min, file_name, folder_path_1min)


# This does not create the 1, 3, 5, 15 min files. But create the pickle files.
async def generate_ohlc_pivot_vi_pickle(db, current_trading_day, previous_trading_day):

    year = current_trading_day.year
    month = current_trading_day.month
    day = current_trading_day.day

    current_dir = os.getcwd()
    pickle_files_dir = os.path.join(current_dir, "pickle_files")

    year = current_trading_day.year
    month = current_trading_day.month
    day = current_trading_day.day

    folder_path_pickle = os.path.join(pickle_files_dir, str(year), str(month), str(day))

    df_stock_master = get_stock_master_data(db)
    build_save_combined_df(df_stock_master, current_trading_day, folder_path_pickle)

    year = previous_trading_day.year
    month = previous_trading_day.month
    day = previous_trading_day.day

    folder_path_pickle = os.path.join(pickle_files_dir, str(year), str(month), str(day))
    folder_path_15min = os.path.join(folder_path_pickle, str(15))
    folder_path_5min = os.path.join(folder_path_pickle, str(5))
    folder_path_3min = os.path.join(folder_path_pickle, str(3))

    file_name = "vi.pkl"
    vi_15 = load_pickle_from_file(file_name, folder_path=folder_path_15min)
    vi_5 = load_pickle_from_file(file_name, folder_path=folder_path_5min)
    vi_3 = load_pickle_from_file(file_name, folder_path=folder_path_3min)

    year = current_trading_day.year
    month = current_trading_day.month
    day = current_trading_day.day

    folder_path_pickle = os.path.join(pickle_files_dir, str(year), str(month), str(day))
    folder_path_15min = os.path.join(folder_path_pickle, str(15))
    folder_path_5min = os.path.join(folder_path_pickle, str(5))
    folder_path_3min = os.path.join(folder_path_pickle, str(3))

    # await call_streaming_vectorized_indicators(
    #     folder_path_15min,
    #     folder_path_5min,
    #     folder_path_3min,
    #     previous_trading_day,
    #     vi_15,
    #     vi_5,
    #     vi_3,
    # )


async def generate_first_day_vectorized_indicators():
    # Load the template with all the stock id, time frame and indicator id
    indicator_data: Dict[int, Dict[int, Dict[int, BaseIndicator]]] = (
        load_dict_from_file("indicator_data.pkl")
    )

    pickle_path_15 = "combined_intraday_data_15.pkl"
    pickle_path_5 = "combined_intraday_data_5.pkl"
    pickle_path_3 = "combined_intraday_data_3.pkl"
    vector_indicators_15 = Vector_Indicators(15, indicator_data)
    vector_indicators_5 = Vector_Indicators(5, indicator_data)
    vector_indicators_3 = Vector_Indicators(3, indicator_data)

    # folder_path=None,
    # folder_path_1min=None,
    # time_interval: int = None,

    # Different time frame tasks can run concurrently
    task_15 = asyncio.create_task(
        process_vector_indicators(vector_indicators_15, pickle_path_15)
    )
    task_5 = asyncio.create_task(
        process_vector_indicators(vector_indicators_5, pickle_path_5)
    )
    task_3 = asyncio.create_task(
        process_vector_indicators(vector_indicators_3, pickle_path_3)
    )

    # Fetch the combined df with all the technical indicators
    combined_df_15, combined_df_5, combined_df_3 = await asyncio.gather(
        task_15, task_5, task_3
    )

    last_trading_day = combined_df_15.index[-1].date()

    combined_df_15 = combined_df_15[combined_df_15.index.date == last_trading_day]
    combined_df_5 = combined_df_5[combined_df_5.index.date == last_trading_day]
    combined_df_3 = combined_df_3[combined_df_3.index.date == last_trading_day]

    save_combined_df_vector_indicators_to_pickle(
        combined_df_15,
        combined_df_5,
        combined_df_3,
        vector_indicators_15,
        vector_indicators_5,
        vector_indicators_3,
    )


def build_save_1min_combined_df(df_stock_master, trading_date, folder_path):

    stock_dfs_15 = {}
    stock_dfs_5 = {}
    stock_dfs_3 = {}
    year = trading_date.year
    month = trading_date.month
    day = trading_date.day

    # Load all stock dataframes
    for _, row in df_stock_master.iterrows():
        stock_id = row["stock_id"]
        stock_symbol = row["stock_symbol"]
        # Later on we do not have intraday data for this stock
        if stock_symbol == "VBL" or stock_symbol == "IDFC" or stock_symbol == "DRREDDY":
            continue

        file_name = f"{stock_id}.csv"
        folder_path_15min = f"historical/{year}/{month}/{day}/15/{stock_id}/"
        folder_path_5min = f"historical/{year}/{month}/{day}/5/{stock_id}/"
        folder_path_3min = f"historical/{year}/{month}/{day}/3/{stock_id}/"

        # if os.path.exists(file_name):
        df_15 = load_csv_to_dataframe(folder_path_15min, file_name)
        df_5 = load_csv_to_dataframe(folder_path_5min, file_name)
        df_3 = load_csv_to_dataframe(folder_path_3min, file_name)

        df_15["datetime"] = df_15["datetime"].apply(parse_datetime)
        # Convert datetime to string for consistent indexing
        # df_15["datetime_str"] = df_15["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
        df_15.set_index("datetime", inplace=True)
        stock_dfs_15[stock_id] = df_15

        df_5["datetime"] = df_5["datetime"].apply(parse_datetime)
        # df_5["datetime_str"] = df_5["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
        df_5.set_index("datetime", inplace=True)
        stock_dfs_5[stock_id] = df_5

        df_3["datetime"] = df_3["datetime"].apply(parse_datetime)
        # df_3["datetime_str"] = df_3["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
        df_3.set_index("datetime", inplace=True)
        stock_dfs_3[stock_id] = df_3

    # Assuming all dataframes have the same datetime range, we can use any of them as reference
    reference_df_15 = next(iter(stock_dfs_15.values()))
    reference_df_5 = next(iter(stock_dfs_5.values()))
    reference_df_3 = next(iter(stock_dfs_3.values()))

    # Prepare data for the combined dataframe
    combined_data_15 = get_combined_df(reference_df_15, stock_dfs_15)
    combined_data_5 = get_combined_df(reference_df_5, stock_dfs_5)
    combined_data_3 = get_combined_df(reference_df_3, stock_dfs_3)

    file_name = "ci_ohlc_pivot.pkl"  # The file name does not change
    # The folder path is different
    folder_path_15min = os.path.join(folder_path, "15")
    folder_path_5min = os.path.join(folder_path, "5")
    folder_path_3min = os.path.join(folder_path, "3")

    save_pickle_to_file(combined_data_15, file_name, folder_path=folder_path_15min)
    save_pickle_to_file(combined_data_5, file_name, folder_path=folder_path_5min)
    save_pickle_to_file(combined_data_3, file_name, folder_path=folder_path_3min)


# After we have converted to dataframe of dataframe indexed by time. We call the technical indicators
async def generate_streaming_vectorized_indicators(
    folder_path_15min,
    folder_path_5min,
    folder_path_3min,
    previous_date=None,
    vi_15=None,
    vi_5=None,
    vi_3=None,
    folder_path_1min=None,
):
    # Load the template with all the stock id, time frame and indicator id
    indicator_data: Dict[int, Dict[int, Dict[int, BaseIndicator]]] = (
        load_dict_from_file("indicator_data.pkl")
    )

    pickle_path_15 = ""
    pickle_path_5 = ""
    pickle_path_3 = ""
    if previous_date is None:
        vector_indicators_15 = Vector_Indicators(15, indicator_data)
        vector_indicators_5 = Vector_Indicators(5, indicator_data)
        vector_indicators_3 = Vector_Indicators(3, indicator_data)
    else:
        vector_indicators_15 = vi_15
        vector_indicators_5 = vi_5
        vector_indicators_3 = vi_3

    # Different time frame tasks can run concurrently
    task_15 = asyncio.create_task(
        process_vector_indicators(
            vector_indicators_15,
            pickle_path_15,
            folder_path_15min,
            folder_path_1min=folder_path_1min,
            time_interval=15,
        )
    )
    task_5 = asyncio.create_task(
        process_vector_indicators(
            vector_indicators_5,
            pickle_path_5,
            folder_path_5min,
            folder_path_1min=folder_path_1min,
            time_interval=5,
        )
    )
    task_3 = asyncio.create_task(
        process_vector_indicators(
            vector_indicators_3,
            pickle_path_3,
            folder_path_3min,
            folder_path_1min=folder_path_1min,
            time_interval=3,
        )
    )

    # Fetch the combined df with all the technical indicators
    combined_df_15, combined_df_5, combined_df_3 = await asyncio.gather(
        task_15, task_5, task_3
    )

    save_combined_df_vector_indicators_to_pickle(
        combined_df_15,
        combined_df_5,
        combined_df_3,
        vector_indicators_15,
        vector_indicators_5,
        vector_indicators_3,
    )
    return vector_indicators_15, vector_indicators_5, vector_indicators_3


async def generate_single_ta(trading_date):
    current_dir = os.getcwd()
    historical_files_dir = os.path.join(current_dir, "historical")
    pickle_files_dir = os.path.join(current_dir, "pickle_files")
    previous_date = None
    vi_15 = None
    vi_5 = None
    vi_3 = None

    year = trading_date.year
    month = trading_date.month
    day = trading_date.day

    # Construct folder path
    folder_path = os.path.join(historical_files_dir, str(year), str(month), str(day))

    # Check if folder exists and then call the method in Vectorized.py
    if os.path.exists(folder_path):
        # We are saving pickle files in a different folder
        folder_path_pickle = os.path.join(
            pickle_files_dir, str(year), str(month), str(day)
        )
        folder_path_15min = os.path.join(folder_path_pickle, "15")
        folder_path_5min = os.path.join(folder_path_pickle, "5")
        folder_path_3min = os.path.join(folder_path_pickle, "3")
        # This is passed to load the combined_df of 1 min for linear regerssion.
        folder_path_1min = os.path.join(folder_path_pickle, "1")

        vector_indicators_15, vector_indicators_5, vector_indicators_3 = (
            await generate_streaming_vectorized_indicators(
                folder_path_15min,
                folder_path_5min,
                folder_path_3min,
                previous_date,
                vi_15,
                vi_5,
                vi_3,
                folder_path_1min,
            )
        )


async def generate_full_ta(
    df_trading_days,
    prior_date=None,
    vi_start_15=None,
    vi_start_5=None,
    vi_start_3=None,
):
    current_dir = os.getcwd()
    historical_files_dir = os.path.join(current_dir, "historical")
    pickle_files_dir = os.path.join(current_dir, "pickle_files")
    previous_date = prior_date
    vi_15 = vi_start_15
    vi_5 = vi_start_5
    vi_3 = vi_start_3

    # Loop through trading days
    for _, row in df_trading_days.iterrows():
        trading_date = row["date"]
        year = trading_date.year
        month = trading_date.month
        day = trading_date.day

        # Construct folder path
        folder_path = os.path.join(
            historical_files_dir, str(year), str(month), str(day)
        )

        # Check if folder exists and then call the method in Vectorized.py
        if os.path.exists(folder_path):
            # We are saving pickle files in a different folder
            folder_path_pickle = os.path.join(
                pickle_files_dir, str(year), str(month), str(day)
            )
            folder_path_15min = os.path.join(folder_path_pickle, "15")
            folder_path_5min = os.path.join(folder_path_pickle, "5")
            folder_path_3min = os.path.join(folder_path_pickle, "3")
            # This is passed to load the combined_df of 1 min for linear regerssion.
            folder_path_1min = os.path.join(folder_path_pickle, "1")

            vector_indicators_15, vector_indicators_5, vector_indicators_3 = (
                await generate_streaming_vectorized_indicators(
                    folder_path_15min,
                    folder_path_5min,
                    folder_path_3min,
                    previous_date,
                    vi_15,
                    vi_5,
                    vi_3,
                    folder_path_1min,
                )
            )
            vi_15 = vector_indicators_15
            vi_5 = vector_indicators_5
            vi_3 = vector_indicators_3

            previous_date = trading_date


# TODO: 1. Need to first save for 28th March 2024. Verify and then do it for 1st April 2024 and then for all entries
async def main():

    # ########### Test revert back to individual stock ohlc and indicator data only for 3 min time frame.
    client = connect_to_mongodb()
    db = client["nse_stock_data"]

    df_trading_days = get_trading_days_data(db, limit=400)

    # # ########### Start of Get the trading days to csv file
    # collection_name = "trading_days"
    # trading_days_col = db[collection_name]
    # df_trading_days = pd.DataFrame(
    #     list(trading_days_col.find().sort("date", 1).limit(400))
    # )
    # df_trading_days.index = df_trading_days.index + 1
    # folder_path = os.path.join(os.getcwd(), "pickle_files/")
    # file_name = "trading_days.csv"
    # file_path = os.path.join(folder_path, file_name)
    # df_trading_days.to_csv(file_path, index_label="index", columns=["date"])
    # return
    # # ########### End of Get the trading days to csv file

    last_day_of_month_df = get_start_streaming_trading_days(df_trading_days)
    df_stock_master = get_stock_master_data(db)

    # last_day_march_2024 = last_day_of_month_df.iloc[0]["date"]
    last_day_march_2024 = pd.to_datetime("2024-12-17")
    year = last_day_march_2024.year
    month = last_day_march_2024.month
    day = last_day_march_2024.day

    file_name = "cid_ta.pkl"
    current_dir = os.getcwd()
    pickle_files_dir = os.path.join(current_dir, "pickle_files")
    folder_path_pickle = os.path.join(pickle_files_dir, str(year), str(month), str(day))
    folder_path = os.path.join(folder_path_pickle, "3")

    combined_df = load_pickle_from_file(file_name, folder_path=folder_path)

    revert_to_stock_df(combined_df, last_day_march_2024, 3)
    client.close()
    return
    # ########### Till here

    # # # ########### Generate pickle file for all indicators for a single day
    # client = connect_to_mongodb()
    # db = client["nse_stock_data"]

    # df_trading_days = get_trading_days_data(db, limit=400)
    # last_day_of_month_df = get_start_streaming_trading_days(df_trading_days)
    # df_stock_master = get_stock_master_data(db)

    # last_day_march_2024 = last_day_of_month_df.iloc[0]["date"]

    # query = {"date": {"$gte": last_day_march_2024}}
    # df_trading_days = get_trading_days_data(db, limit=400, query=query)

    # trading_date = df_trading_days.iloc[-1]["date"]
    # await generate_single_ta(last_day_march_2024)

    # client.close()
    # return
    # # ################# Till here

    # # # ########### Generate pickle file for all indicators from the start
    # client = connect_to_mongodb()
    # db = client["nse_stock_data"]

    # df_trading_days = get_trading_days_data(db, limit=400)
    # last_day_of_month_df = get_start_streaming_trading_days(df_trading_days)
    # df_stock_master = get_stock_master_data(db)

    # last_day_march_2024 = last_day_of_month_df.iloc[0]["date"]
    # last_day_march_2024 = pd.to_datetime("2024-06-11")
    # year = last_day_march_2024.year
    # month = last_day_march_2024.month
    # day = last_day_march_2024.day

    # query = {"date": {"$gt": last_day_march_2024}}
    # df_trading_days = get_trading_days_data(db, limit=400, query=query)

    # file_name = "vi.pkl"
    # current_dir = os.getcwd()
    # pickle_files_dir = os.path.join(current_dir, "pickle_files")

    # folder_path_pickle = os.path.join(pickle_files_dir, str(year), str(month), str(day))

    # # Load pickle files from the previous trading folder and continue
    # folder_path = os.path.join(folder_path_pickle, "3")
    # vi_start_3 = load_pickle_from_file(file_name, folder_path=folder_path)

    # folder_path = os.path.join(folder_path_pickle, "5")
    # vi_start_5 = load_pickle_from_file(file_name, folder_path=folder_path)

    # folder_path = os.path.join(folder_path_pickle, "15")
    # vi_start_15 = load_pickle_from_file(file_name, folder_path=folder_path)

    # await generate_full_ta(
    #     df_trading_days, last_day_march_2024, vi_start_15, vi_start_5, vi_start_3
    # )

    # client.close()
    # return
    # # ################# Till here

    # for index, row in df_trading_days.iterrows():
    # trading_date = row["date"]
    # year = trading_date.year
    # month = trading_date.month
    # day = trading_date.day

    # folder_path = os.path.join(os.getcwd(), "pickle_files/")
    # folder_path_pickle = os.path.join(folder_path, str(year), str(month), str(day))

    # return
    # df_trading_days = get_trading_days_data(db, limit=2)
    # trading_date = df_trading_days.iloc[0]["date"]
    # await generate_ohlc_daily_data(db, last_day_march_2024, first_day_march_2024)
    # await generate_ohlc_daily_data(db, trading_date)
    # build_save_1min_combined_df(df_stock_master, trading_date, folder_path_pickle)

    # df_trading_days = get_trading_days_data(db, limit=400)
    # last_day_of_month_df = get_start_streaming_trading_days(df_trading_days)
    # df_stock_master = get_stock_master_data(db)

    # last_day_march_2024 = last_day_of_month_df.iloc[0]["date"]

    # for index, row in df_trading_days.iterrows():
    #     trading_date = row["date"]
    #     await generate_ohlc_daily_data(db, trading_date)

    # start_date_5min = pd.Timestamp(last_day_of_month_df.iloc[6]["date"]).date()
    # start_date_3min = pd.Timestamp(last_day_of_month_df.iloc[3]["date"]).date()
    # print(last_day_march_2024, start_date_5min, start_date_3min)
    # query = {"date": {"$gte": last_day_march_2024}}
    # df_trading_days = get_trading_days_data(db, limit=400, query=query)

    # create_folders(df_trading_days)
    # start_date_15min = df_trading_days.iloc[0]["date"]
    # loop_market_data_generate_df_save_csv(
    #     db, market_data, start_date_15min, start_date_5min, start_date_3min
    # )

    previous_date = pd.to_datetime("2024-10-25")
    start_streaming_date = pd.to_datetime("2024-10-28")

    folder_path = os.path.join(os.getcwd(), "pickle_files/")
    file_name = "trading_days.csv"

    # The problem of using this is that trading_days.csv may not be upto date
    df_trading_days = load_csv_to_dataframe(folder_path, file_name=file_name)

    await reset_swing_levels_from(df_trading_days, start_streaming_date, previous_date)
    # build_save_combined_df(df_stock_master, start_streaming_trading_day)
    # print(start_streaming_trading_day)

    # client.close()
    # return
    # with timer("call_streaming_vectorized_indicators"):
    #     await call_streaming_vectorized_indicators()
    # 1. After doing this we would have to unpack the pickle file and view the head and tail of the dataframe.
    # 2. Do not spend time on replicating this for all of 8 months. First complete the event notification.
    # 3. We can do the same for the latest trading day as we have done for start of streaming day.
    # 4. So that we can test the next day using live data.

    # # We have the prior trading days file name and folder path for a vectors indicators pickle file and we continue from there.
    # vector_indicators = load_pickle_from_file(
    #     vector_indicators_pickle_file_name,
    #     folder_path=vector_indicators_pickle_folder,
    # )
    # # Verify that the pickle file has loaded correctly and is of the expected type
    # if isinstance(vector_indicators, Vector_Indicators):
    #     # If pickle file has loaded correctly then assign it to indicator data
    #     vector_indicators: Vector_Indicators = vector_indicators


if __name__ == "__main__":
    asyncio.run(main())
    # main()

# After we have converted to dataframe of dataframe indexed by time. We call the technical indicators
# async def generate_first_day_vectorized_indicators():
#     # Load the template with all the stock id, time frame and indicator id
#     indicator_data: Dict[int, Dict[int, Dict[int, BaseIndicator]]] = (
#         load_dict_from_file("indicator_data.pkl")
#     )

#     pickle_path_15 = "combined_intraday_data_15.pkl"
#     pickle_path_5 = "combined_intraday_data_5.pkl"
#     pickle_path_3 = "combined_intraday_data_3.pkl"
#     vector_indicators_15 = Vector_Indicators(15, indicator_data)
#     vector_indicators_5 = Vector_Indicators(5, indicator_data)
#     vector_indicators_3 = Vector_Indicators(3, indicator_data)

#     # Different time frame tasks can run concurrently
#     task_15 = asyncio.create_task(
#         process_vector_indicators(vector_indicators_15, pickle_path_15)
#     )
#     task_5 = asyncio.create_task(
#         process_vector_indicators(vector_indicators_5, pickle_path_5)
#     )
#     task_3 = asyncio.create_task(
#         process_vector_indicators(vector_indicators_3, pickle_path_3)
#     )

#     # Fetch the combined df with all the technical indicators
#     combined_df_15, combined_df_5, combined_df_3 = await asyncio.gather(
#         task_15, task_5, task_3
#     )

#     last_trading_day = combined_df_15.index[-1].date()

#     # save_vector_data_indicators_csv(vector_indicators_15, last_trading_day, 15)
#     # save_vector_data_indicators_csv(vector_indicators_5, last_trading_day, 5)
#     # save_vector_data_indicators_csv(vector_indicators_3, last_trading_day, 3)
#     # return

#     combined_df_15 = combined_df_15[combined_df_15.index.date == last_trading_day]
#     combined_df_5 = combined_df_5[combined_df_5.index.date == last_trading_day]
#     combined_df_3 = combined_df_3[combined_df_3.index.date == last_trading_day]

#     # # This is to verify if streaming technical indicators map with the one shot technical indicators.
#     # revert_to_stock_df(combined_df_15, last_trading_day, 15)
#     # revert_to_stock_df(combined_df_5, last_trading_day, 5)
#     # revert_to_stock_df(combined_df_3, last_trading_day, 3)
#     # return

#     save_combined_df_vector_indicators_to_pickle(
#         combined_df_15,
#         combined_df_5,
#         combined_df_3,
#         vector_indicators_15,
#         vector_indicators_5,
#         vector_indicators_3,
#     )
