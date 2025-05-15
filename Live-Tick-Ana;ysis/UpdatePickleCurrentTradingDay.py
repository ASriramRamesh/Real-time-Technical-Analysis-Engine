import asyncio

import pandas as pd
import numpy as np
from pymongo import MongoClient
from typing import Dict
import pytz
from datetime import datetime, timedelta
import sqlite3
import pickle
import os
import sys

from QueryIndicators import (
    connect_to_mongodb,
    get_trading_days_data,
    resample_stock_data,
    save_to_csv,
    get_stock_master_data,
)

from SaveStreaming import load_dict_from_file

from Vectorized import (
    build_save_combined_df,
    load_pickle_from_file,
    generate_streaming_vectorized_indicators,
    save_vector_data_indicators_csv,
)

from BuildTradingDaysFolders import (
    get_start_streaming_trading_days,
    get_stock_data_till,
)

from BuildPickleFiles import call_streaming_vectorized_indicators


async def generate_ohlc_pivot_pickle_files(
    db, current_trading_day, previous_trading_day
):

    next_date = current_trading_day + timedelta(days=1)
    year = current_trading_day.year
    month = current_trading_day.month
    day = current_trading_day.day

    market_data = load_dict_from_file("market_data.pkl")
    for stock_id, stock in market_data.items():
        stock_symbol = stock["symbol"]

        file_name = f"{stock_id}.csv"
        folder_path_15min = f"historical/{year}/{month}/{day}/15/{stock_id}/"
        folder_path_5min = f"historical/{year}/{month}/{day}/5/{stock_id}/"
        folder_path_3min = f"historical/{year}/{month}/{day}/3/{stock_id}/"
        folder_path_1min = f"historical/{year}/{month}/{day}/1/{stock_id}/"
        # We do not have intraday data for this stock hence need to comment
        if stock_symbol == "VBL" or stock_symbol == "IDFC" or stock_symbol == "DRREDDY":
            continue
        # Fetches the stock data for that date range.

        df_1min = get_stock_data_till(db, stock_symbol, next_date, current_trading_day)
        if df_1min is None:
            continue
        df_15min = resample_stock_data(df_1min)
        df_5min = resample_stock_data(df_1min, period="5min")
        df_3min = resample_stock_data(df_1min, period="3min")
        # Save to appropriate folder
        save_to_csv(df_15min, file_name, folder_path_15min)
        save_to_csv(df_5min, file_name, folder_path_5min)
        save_to_csv(df_3min, file_name, folder_path_3min)
        save_to_csv(df_1min, file_name, folder_path_1min)

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
    # This is passed to load the combined_df of 1 min for linear regerssion.
    folder_path_1min = os.path.join(folder_path_pickle, "1")

    await generate_streaming_vectorized_indicators(
        folder_path_15min,
        folder_path_5min,
        folder_path_3min,
        previous_date=previous_trading_day,
        vi_15=vi_15,
        vi_5=vi_5,
        vi_3=vi_3,
        folder_path_1min=folder_path_1min,
    )


async def main():

    client = connect_to_mongodb()
    db = client["nse_stock_data"]

    df_trading_days = get_trading_days_data(db, limit=2)
    current_trading_day = df_trading_days.iloc[0]["date"]
    previous_trading_day = df_trading_days.iloc[-1]["date"]
    # # ######## Generate swing level csv files for the current trading day starts here
    # current_dir = os.getcwd()
    # pickle_files_dir = os.path.join(current_dir, "pickle_files")
    # year = current_trading_day.year
    # month = current_trading_day.month
    # day = current_trading_day.day

    # folder_path_pickle = os.path.join(pickle_files_dir, str(year), str(month), str(day))
    # folder_path_15min = os.path.join(folder_path_pickle, "15")
    # folder_path_5min = os.path.join(folder_path_pickle, "5")
    # folder_path_3min = os.path.join(folder_path_pickle, "3")

    # file_name = "vi.pkl"
    # vi_3 = load_pickle_from_file(file_name, folder_path=folder_path_3min)
    # vi_5 = load_pickle_from_file(file_name, folder_path=folder_path_5min)
    # vi_15 = load_pickle_from_file(file_name, folder_path=folder_path_15min)

    # save_vector_data_indicators_csv(vi_15, current_trading_day, 15)
    # save_vector_data_indicators_csv(vi_5, current_trading_day, 5)
    # save_vector_data_indicators_csv(vi_3, current_trading_day, 3)

    # client.close()
    # return
    # # ######## Generate swing level csv files ends here

    await generate_ohlc_pivot_pickle_files(
        db, current_trading_day, previous_trading_day
    )

    client.close()
    return
    # ########## If we want to enter for a particular date then first verify and then call generate
    # stock_symbol = "Nifty_50"
    # collection_name = f"stock_{stock_symbol}"
    # stock_col = db[collection_name]

    # query = {"datetime": {"$gte": current_trading_day, "$lte": next_date}}
    # df = pd.DataFrame(list(stock_col.find(query).sort("datetime", 1)))

    # df["datetime"] = df["datetime"].dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
    # df.set_index("datetime", inplace=True)

    # print(df.head())
    # print(df.tail())
    # return
    # df_1min = get_stock_data_till(
    #     db, "Nifty_50", current_trading_day, current_trading_day
    # )
    # print(df_1min.head())
    # print(df_1min.tail())
    # return
    # print("current_trading_day", current_trading_day)
    # print("previous_trading_day", previous_trading_day)
    # client.close()
    # return
    # df_trading_days = get_trading_days_data(db, limit=400)
    # last_day_of_month_df = get_start_streaming_trading_days(df_trading_days)

    # last_day_march_2024 = last_day_of_month_df.iloc[0]["date"]

    # query = {"date": {"$gte": last_day_march_2024}}
    # df_trading_days = get_trading_days_data(db, limit=2, query=query)
    # previous_trading_day = df_trading_days.iloc[0]["date"]
    # current_trading_day = df_trading_days.iloc[-1]["date"]


if __name__ == "__main__":
    asyncio.run(main())
