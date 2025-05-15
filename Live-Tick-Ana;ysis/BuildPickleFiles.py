import asyncio
from typing import Dict

import pandas as pd
import numpy as np
from pymongo import MongoClient
import os
import shutil
from datetime import datetime, timedelta
import time
from contextlib import contextmanager

from Vectorized import (
    build_save_combined_df,
    save_combined_df_vector_indicators_to_pickle,
    process_vector_indicators,
    load_pickle_from_file,
    save_vector_data_indicators_csv,
    revert_to_stock_df,
)

from ta.event_indicators import Vector_Indicators, BaseIndicator, SwingLevel

from QueryIndicators import (
    connect_to_mongodb,
    get_trading_days_data,
    resample_stock_data,
    save_to_csv,
    load_csv_to_dataframe,
    calculate_daily_levels,
    calculate_supertrend_levels,
    create_empty_timeseries_dataframes,
    get_stock_master_data,
)
from SaveStreaming import load_dict_from_file
from BuildTradingDaysFolders import (
    get_start_streaming_trading_days,
    retrieve_market_data,
    loop_market_data,
    generate_daily_time_frame_data,
)


@contextmanager
def timer(name):
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    print(f"{name}: {end_time - start_time:.4f} seconds")


# This returns the days for which we have up loaded intraday data.
def get_trading_days_data_ascending(db, limit=10):
    collection_name = "trading_days"
    trading_days_col = db[collection_name]
    return pd.DataFrame(list(trading_days_col.find().sort("date", 1).limit(limit)))


# The pickle files do not need stock_id folders so remove them for cleaner file structure.
def delete_stock_folder_files(df_trading_days):
    # Define paths
    current_dir = os.getcwd()
    pickle_files_dir = os.path.join(current_dir, "pickle_files")

    # Loop through trading days
    for index, row in df_trading_days.iterrows():
        date = row["date"]
        year = date.year
        month = date.month
        day = date.day

        # Construct folder path
        folder_path = os.path.join(pickle_files_dir, str(year), str(month), str(day))

        # Check if folder exists
        if os.path.exists(folder_path):
            # Define subfolder paths
            subfolders = [os.path.join(folder_path, str(i)) for i in [3, 5, 15]]

            # Loop through subfolders
            for subfolder in subfolders:
                if os.path.exists(subfolder):
                    # Delete all files and folders inside subfolder
                    for filename in os.listdir(subfolder):
                        file_path = os.path.join(subfolder, filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)


def revert_back_stock_ohlc_ta_csv():
    current_dir = os.getcwd()
    pickle_files_dir = os.path.join(current_dir, "pickle_files")

    last_trading_day = pd.to_datetime("2024-11-14")
    year = last_trading_day.year
    month = last_trading_day.month
    day = last_trading_day.day

    folder_path_pickle = os.path.join(pickle_files_dir, str(year), str(month), str(day))
    folder_path_15min = os.path.join(folder_path_pickle, str(15))
    folder_path_5min = os.path.join(folder_path_pickle, str(5))
    folder_path_3min = os.path.join(folder_path_pickle, str(3))

    file_name = "vi.pkl"
    vi_15 = load_pickle_from_file(file_name, folder_path=folder_path_15min)
    vi_5 = load_pickle_from_file(file_name, folder_path=folder_path_5min)
    vi_3 = load_pickle_from_file(file_name, folder_path=folder_path_3min)

    save_vector_data_indicators_csv(vi_15, last_trading_day, 15)
    save_vector_data_indicators_csv(vi_5, last_trading_day, 5)
    save_vector_data_indicators_csv(vi_3, last_trading_day, 3)
    return

    file_name = "cid_ta.pkl"
    combined_df_15 = load_pickle_from_file(file_name, folder_path=folder_path_15min)
    combined_df_5 = load_pickle_from_file(file_name, folder_path=folder_path_5min)
    combined_df_3 = load_pickle_from_file(file_name, folder_path=folder_path_3min)

    revert_to_stock_df(combined_df_15, last_trading_day, 15)
    revert_to_stock_df(combined_df_5, last_trading_day, 5)
    revert_to_stock_df(combined_df_3, last_trading_day, 3)


def generate_OHLC_data_from(start_date):
    start_streaming_date = pd.to_datetime(start_date)
    folder_path = os.path.join(os.getcwd(), "pickle_files/")
    file_name = "trading_days.csv"

    df_trading_days = load_csv_to_dataframe(folder_path, file_name=file_name)
    df_trading_days["date"] = pd.to_datetime(df_trading_days["date"])
    filtered_df = df_trading_days[df_trading_days["date"] > start_date]

    client = connect_to_mongodb()
    db = client["nse_stock_data"]

    generate_daily_time_frame_data(db, filtered_df, start_streaming_date)
    client.close()


def generate_pivoted_ohlc_based_on_datetime(df_trading_days, df_stock_master):
    current_dir = os.getcwd()
    historical_files_dir = os.path.join(current_dir, "historical")
    pickle_files_dir = os.path.join(current_dir, "pickle_files")

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

            build_save_combined_df(df_stock_master, trading_date, folder_path_pickle)
            # We will verify that it works properly for the first record and then loop for the entire data.


# After we have converted to dataframe of dataframe indexed by time. We call the technical indicators
async def call_streaming_vectorized_indicators(
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


async def generate_pivoted_full_ta_based_on_datetime(df_trading_days, df_stock_master):
    current_dir = os.getcwd()
    historical_files_dir = os.path.join(current_dir, "historical")
    pickle_files_dir = os.path.join(current_dir, "pickle_files")
    previous_date = None
    vi_15 = None
    vi_5 = None
    vi_3 = None

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

            vector_indicators_15, vector_indicators_5, vector_indicators_3 = (
                await call_streaming_vectorized_indicators(
                    folder_path_15min,
                    folder_path_5min,
                    folder_path_3min,
                    previous_date,
                    vi_15,
                    vi_5,
                    vi_3,
                )
            )
            vi_15 = vector_indicators_15
            vi_5 = vector_indicators_5
            vi_3 = vector_indicators_3

            previous_date = trading_date


async def main():

    # revert_back_stock_ohlc_ta_csv()
    # return

    # Load the list of stocks
    folder_path = os.path.join(os.getcwd(), "pickle_files/")
    file_name = "stock_master.csv"

    df_stock_master = load_csv_to_dataframe(folder_path, file_name=file_name)

    # Load the list of trading days
    folder_path = os.path.join(os.getcwd(), "pickle_files/")
    file_name = "trading_days.csv"

    df_trading_days = load_csv_to_dataframe(folder_path, file_name=file_name)
    df_trading_days["date"] = pd.to_datetime(df_trading_days["date"])
    await generate_pivoted_full_ta_based_on_datetime(df_trading_days, df_stock_master)

    # revert_back_stock_ohlc_ta_csv()
    # return
    # current_dir = os.getcwd()
    # pickle_files_dir = os.path.join(current_dir, "pickle_files")
    # previous_date = pd.to_datetime("2024-03-28")
    # year = previous_date.year
    # month = previous_date.month
    # day = previous_date.day

    # folder_path_pickle = os.path.join(pickle_files_dir, str(year), str(month), str(day))
    # folder_path_15min = os.path.join(folder_path_pickle, str(15))
    # folder_path_5min = os.path.join(folder_path_pickle, str(5))
    # folder_path_3min = os.path.join(folder_path_pickle, str(3))

    # file_name = "vi.pkl"
    # vi_15 = load_pickle_from_file(file_name, folder_path=folder_path_15min)
    # vi_5 = load_pickle_from_file(file_name, folder_path=folder_path_5min)
    # vi_3 = load_pickle_from_file(file_name, folder_path=folder_path_3min)

    # start_streaming_date = pd.to_datetime("2024-04-01")
    # year = start_streaming_date.year
    # month = start_streaming_date.month
    # day = start_streaming_date.day

    # folder_path_pickle = os.path.join(pickle_files_dir, str(year), str(month), str(day))
    # folder_path_15min = os.path.join(folder_path_pickle, str(15))
    # folder_path_5min = os.path.join(folder_path_pickle, str(5))
    # folder_path_3min = os.path.join(folder_path_pickle, str(3))

    # await call_streaming_vectorized_indicators(
    #     folder_path_15min,
    #     folder_path_5min,
    #     folder_path_3min,
    #     previous_date,
    #     vi_15,
    #     vi_5,
    #     vi_3,
    # )
    # return
    # # We would have to write a method that would update the current date to the trading_days.csv file
    # start_date = "2024-10-24"
    # generate_OHLC_data_from(start_date)
    # return

    # filtered_df = df_trading_days[df_trading_days["date"] > "2024-10-21"]
    # generate_pivoted_ohlc_based_on_datetime(filtered_df, df_stock_master)


if __name__ == "__main__":
    asyncio.run(main())
