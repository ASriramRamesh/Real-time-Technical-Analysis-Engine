from ta.pandas_ta import (
    rsi as ta_rsi,
    supertrend as ta_supertrend,
    ema as ta_ema,
    ppo as ta_ppo,
    roc as ta_roc,
)
from ta.streaming_indicators import (
    RSI as streaming_rsi,
    SuperTrend as streaming_supertrend,
    EMA as streaming_ema,
    PPO as streaming_ppo,
    ROC as streaming_roc,
)

import pandas as pd
import numpy as np
from pymongo import MongoClient
import os
from datetime import datetime, timedelta
import time
from contextlib import contextmanager

from QueryIndicators import (
    connect_to_mongodb,
    get_trading_days_data,
    resample_stock_data,
    save_to_csv,
    load_csv_to_dataframe,
    calculate_daily_levels,
    calculate_supertrend_levels,
    create_empty_timeseries_dataframes,
)
from SaveStreaming import load_dict_from_file

from SuperTrendLevels import SwingLevel

HLC3 = "hlc3"
supertrend_first_period_value = "st_fpvalue"
supertrend_first_period_direction = "st_fpdirection"
supertrend_second_period_value = "st_spvalue"
supertrend_second_period_direction = "st_spdirection"
EMA9_column_name = "ema_9"
EMA21_column_name = "ema_21"
RSI_column_name = "rsi_14"
Smoothed_RSI_column_name = "ema_rsi"
ppo_hlc3_ema9_column_name = "ppo_hlc3ema9"
ppo_ema9_ema21_column_name = "ppo_ema9ema21"
ppo_rsi_emarsi_column_name = "ppo_rsiema14"
roc_hlc3_column_name = "roc_hlc3"
roc_ema9_column_name = "roc_ema9"
roc_ema21_column_name = "roc_ema21"
roc_rsi_column_name = "roc_rsi"
roc_smoothed_rsi_column_name = "roc_emarsi"

# These are where the indicator pickle files would be loaded to
# The file name would be the key value and ".pkl"
# The folder would be year/month/day/time_frame/stock_id/ema9.pkl
# In the market_data it is market_data[stock_id][time_frame]["indicators"]["ema9"]
# When we start trading we would load the pickle file from the previous day file location
# We will load the pickle file to market_data location and we update it at every time frame and save it back.
# market_data is a global variable which holds the indicator class objects and after time interval.
# We will access the indicator class and give the latest ohlc values and the get the indicator values.
# We will have a ema9 where the column would be the time interval and the index would be stock_id and
# the values would be ema9 value for a stock and time interval.
# We have 8 indicators and 3 time frames. So in all we have 24 data frames.
# When we are doing a study on the stock we could collate the sectors that it belongs to and then
# do a comparison.
ema9_pickle = "ema9"
ema21_pickle = "ema21"
rsi14_pickle = "rsi14"
st_fp_pickle = "st_fp"
st_sp_pickle = "st_sp"
smoothed_rsi_pickle = "smoothed_rsi"
ppo_hlc3_pickle = "ppo_hlc3"
ppo_ema9_pickle = "ppo_ema9"
ppo_rsi_pickle = "ppo_rsi"
roc_hlc3_pickle = "roc_hlc3"
roc_ema9_pickle = "roc_ema9"
roc_ema21_pickle = "roc_ema21"
roc_rsi_pickle = "roc_rsi"
roc_smoothedrsi_pickle = "roc_smoothedrsi"
swing_levels_pickle = "swing_levels"


@contextmanager
def timer(name):
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    print(f"{name}: {end_time - start_time:.4f} seconds")


# Get the start month where we have 1 min data
def get_start_streaming_trading_days(df_trading_days):
    last_row = df_trading_days.iloc[-1]
    last_date = last_row["date"]  # Assuming 'date' is the column name
    last_date_month = last_date.month
    last_date_year = last_date.year

    # Get the last day of the month
    last_day_of_month = pd.to_datetime(
        f"{last_date_year}-{last_date_month}-01"
    ) + pd.DateOffset(months=1, days=-1)

    # Filter the DataFrame to get the last day of the month
    last_day_of_month_df = df_trading_days[
        (df_trading_days["date"].dt.month == last_date_month)
        & (df_trading_days["date"].dt.year == last_date_year)
        # The trading day less than or equal to last day of the month
        & (df_trading_days["date"].dt.day <= last_day_of_month.day)
    ].copy()

    return last_day_of_month_df


# For all the trading days create folder in the month and year and then folder for timeframe 3,5,15 and
# in each time frame folder create the folders from 1 to 99 representing the stock id.
def create_folders(df_trading_days, start_streaming_trading_day):
    for index, row in df_trading_days.iterrows():
        date = row["date"]
        if date > start_streaming_trading_day:
            month = date.month
            day = date.day
            folder_path = f"historical/2024/{month}/{day}"
            os.makedirs(folder_path, exist_ok=True)
            for subfolder in [1]:
                subfolder_path = f"{folder_path}/{subfolder}"
                os.makedirs(subfolder_path, exist_ok=True)
                for i in range(1, 100):
                    subsubfolder_path = f"{subfolder_path}/{i}"
                    os.makedirs(subsubfolder_path, exist_ok=True)


def get_stock_data_till(db, stock_symbol, end_date, start_date=None):
    collection_name = f"stock_{stock_symbol}"
    stock_col = db[collection_name]
    # For comparison we have to change it to datetime.date and for mongodb it has to be Timestamp
    start_date_timestamp = pd.Timestamp(start_date).date()
    end_date_timestamp = pd.Timestamp(end_date).date()

    # Validation
    if start_date is not None:
        if start_date_timestamp > end_date_timestamp:
            print("Error: Start date cannot be greater than end date.")
            return None
        elif start_date_timestamp == end_date_timestamp:
            query = {"datetime": start_date}
        else:
            query = {"datetime": {"$gte": start_date, "$lte": end_date}}
    else:
        query = {"datetime": {"$lte": end_date}}

    df = pd.DataFrame(list(stock_col.find(query).sort("datetime", 1)))

    # Convert datetime to Asia/Kolkata timezone
    if "datetime" in df.columns:
        df["datetime"] = (
            df["datetime"].dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
        )
        df.set_index("datetime", inplace=True)

        return df
    else:
        print(f"Empty dataset for stock symbol: {stock_symbol}")
        return None


def generate_daily_time_frame_data(db, df, start_streaming_date):
    df = df.sort_values(by="date")
    for index, row in df.iterrows():
        current_date = row["date"]
        # Only add data after current day
        if current_date > start_streaming_date:
            next_date = current_date + timedelta(days=1)
            year = current_date.year
            month = current_date.month
            day = current_date.day

            market_data = load_dict_from_file("market_data.pkl")
            for stock_id, stock in market_data.items():
                stock_symbol = stock["symbol"]

                file_name = f"{stock_id}.csv"
                folder_path_15min = f"historical/{year}/{month}/{day}/15/{stock_id}/"
                folder_path_5min = f"historical/{year}/{month}/{day}/5/{stock_id}/"
                folder_path_3min = f"historical/{year}/{month}/{day}/3/{stock_id}/"
                # We do not have intraday data for this stock hence need to comment
                if (
                    stock_symbol == "VBL"
                    or stock_symbol == "IDFC"
                    or stock_symbol == "DRREDDY"
                ):
                    continue
                # Fetches the stock data for that date range.
                df_1min = get_stock_data_till(db, stock_symbol, next_date, current_date)
                df_15min = resample_stock_data(df_1min)
                df_5min = resample_stock_data(df_1min, period="5min")
                df_3min = resample_stock_data(df_1min, period="3min")
                # Save to appropriate folder
                save_to_csv(df_15min, file_name, folder_path_15min)
                save_to_csv(df_5min, file_name, folder_path_5min)
                save_to_csv(df_3min, file_name, folder_path_3min)


def generate_streaming_start_trading_day(df, db):
    # Since df_trading_days is newest first we get the last trading day of month
    # But this would be at 00:00 so we have to add a day to the start streaming trading days variable
    start_streaming_trading_day = df.iloc[0]["date"] + timedelta(days=1)
    start_date_15min = df.iloc[-1]["date"]
    start_date_5min = pd.Timestamp(df.iloc[5]["date"]).date()
    start_date_3min = pd.Timestamp(df.iloc[3]["date"]).date()

    market_data = load_dict_from_file("market_data.pkl")

    for stock_id, stock in market_data.items():
        stock_symbol = stock["symbol"]
        # We do not have intraday data for this stock hence need to comment
        if stock_symbol == "VBL" or stock_symbol == "IDFC" or stock_symbol == "DRREDDY":
            continue

        df_15min_1min = get_stock_data_till(
            db, stock_symbol, start_streaming_trading_day, start_date_15min
        )
        df_15min = resample_stock_data(df_15min_1min)

        df_5min_1min = df_15min_1min[df_15min_1min.index.date >= start_date_5min]
        df_5min = resample_stock_data(df_5min_1min, period="5min")

        df_3min_1min = df_15min_1min[df_15min_1min.index.date >= start_date_3min]
        df_3min = resample_stock_data(df_3min_1min, period="3min")

        file_name = f"{stock_id}.csv"
        folder_path_15min = f"historical/2024/3/28/15/{stock_id}/"
        folder_path_5min = f"historical/2024/3/28/5/{stock_id}/"
        folder_path_3min = f"historical/2024/3/28/3/{stock_id}/"

        save_to_csv(df_15min, file_name, folder_path_15min)
        save_to_csv(df_5min, file_name, folder_path_5min)
        save_to_csv(df_3min, file_name, folder_path_3min)


# When we are looking to vectorize and call indicators for all stocks in one go.
# This has changed the data format to streaming just for 3 min time frame.
# When we switch to live streaming data this is the format we would have data.
# We would have to create a dict object for time_frame/stock_id/indicator_type and store this class.
def create_streaming_market_data(market_data):
    """
    Create a dictionary with time as key and OHLC dataframe as value.

    Parameters:
    stock_data (dict): Dictionary with stock_id as key and pandas timeseries dataframe as value.
    interval (int): Interval in minutes.

    Returns:
    dict: Dictionary with time as key and OHLC dataframe as value.
    """
    interval = 3
    start_time = pd.to_datetime("09:15")
    end_time = pd.to_datetime("15:27")  # adjust end time based on interval
    time_range = pd.date_range(start=start_time, end=end_time, freq=f"{interval}min")

    ohlc_dict = {}
    for time in time_range:
        df = pd.DataFrame(index=market_data.keys())
        for stock_id, data in market_data.items():
            df.loc[stock_id, ["open", "high", "low", "close"]] = data.loc[time][
                ["open", "high", "low", "close"]
            ]
        ohlc_dict[time] = df

    return ohlc_dict


def retrieve_market_data(trading_date: datetime = None):
    market_data = load_dict_from_file("market_data.pkl")

    year = trading_date.year
    month = trading_date.month
    day = trading_date.day

    for stock_id, stock in market_data.items():

        # We do not have intraday data for this stock hence need to comment
        if stock["symbol"] == "VBL" or stock["symbol"] == "IDFC":
            continue

        # Build folder path based on year, month and day
        folder_path_15min = f"historical/{year}/{month}/{day}/15/{stock_id}/"
        folder_path_5min = f"historical/{year}/{month}/{day}/5/{stock_id}/"
        folder_path_3min = f"historical/{year}/{month}/{day}/3/{stock_id}/"

        file_name = f"{stock_id}.csv"

        # Load the csv file for the stock into the dataframe.
        df_15min = load_csv_to_dataframe(
            folder_path_15min, file_name, is_date_index=True
        )
        df_5min = load_csv_to_dataframe(folder_path_5min, file_name, is_date_index=True)
        df_3min = load_csv_to_dataframe(folder_path_3min, file_name, is_date_index=True)

        # Add the computed column hlc3
        df_15min[HLC3] = (
            (df_15min["high"] + df_15min["low"] + df_15min["close"]) / 3
        ).round(2)
        df_5min[HLC3] = (
            (df_5min["high"] + df_5min["low"] + df_5min["close"]) / 3
        ).round(2)
        df_3min[HLC3] = (
            (df_3min["high"] + df_3min["low"] + df_3min["close"]) / 3
        ).round(2)

        stock[15]["ohlc"] = df_15min
        stock[5]["ohlc"] = df_5min
        stock[3]["ohlc"] = df_3min

    return market_data


def save_streaming_indicators(df, stock_id, time_frame, trading_date: datetime = None):

    year = trading_date.year
    month = trading_date.month
    day = trading_date.day
    folder_path = f"historical/{year}/{month}/{day}/{time_frame}/{stock_id}/"

    supertrend_period_length = 10

    if time_frame == 3:
        supertrend_first_period_multiplier = 3
        supertrend_second_period_multiplier = 4
    elif time_frame == 5:
        supertrend_first_period_multiplier = 2
        supertrend_second_period_multiplier = 3
    else:
        supertrend_first_period_multiplier = 1.5
        supertrend_second_period_multiplier = 2.5

    ema9 = streaming_ema(9)
    ema21 = streaming_ema(21)
    rsi14 = streaming_rsi(14)
    st_fp = streaming_supertrend(
        supertrend_period_length, supertrend_first_period_multiplier
    )
    st_sp = streaming_supertrend(
        supertrend_period_length, supertrend_second_period_multiplier
    )
    smoothed_rsi = streaming_ema(14)

    ppo_hlc3 = streaming_ppo()
    ppo_ema9 = streaming_ppo()
    ppo_rsi = streaming_ppo()

    roc_hlc3 = streaming_roc()
    roc_ema9 = streaming_roc()
    roc_ema21 = streaming_roc()
    roc_rsi = streaming_roc()
    roc_smoothedrsi = streaming_roc()

    smoothed_rsi_value = None
    ppo_hlc3_value = None
    ppo_ema9_value = None
    ppo_rsi_value = None

    roc_hlc3_value = None
    roc_ema9_value = None
    roc_ema21_value = None
    roc_rsi_value = None
    roc_smoothedrsi_value = None

    df_streaming = df.copy()
    # Add columns to the streaming dataframe this is for 28th March 2024
    df_streaming[supertrend_first_period_value] = 0.0
    df_streaming[supertrend_first_period_direction] = 0.0
    df_streaming[supertrend_second_period_value] = 0.0
    df_streaming[supertrend_second_period_direction] = 0.0
    df_streaming[EMA9_column_name] = 0.0
    df_streaming[EMA21_column_name] = 0.0
    df_streaming[RSI_column_name] = 0.0
    df_streaming[Smoothed_RSI_column_name] = 0.0
    df_streaming[ppo_hlc3_ema9_column_name] = 0.0
    df_streaming[ppo_ema9_ema21_column_name] = 0.0
    df_streaming[ppo_rsi_emarsi_column_name] = 0.0
    df_streaming[roc_hlc3_column_name] = 0.0
    df_streaming[roc_ema9_column_name] = 0.0
    df_streaming[roc_ema21_column_name] = 0.0
    df_streaming[roc_rsi_column_name] = 0.0
    df_streaming[roc_smoothed_rsi_column_name] = 0.0

    # We keep updating the values for each row of OHLC
    for datetime_index, candle in df.iterrows():
        # We have to update the datetime also
        ema9_value = ema9.update(candle[HLC3], datetime_index)
        df_streaming.at[datetime_index, EMA9_column_name] = np.float64(
            round(ema9_value or 0, 2)
        )

        ema21_value = ema21.update(candle[HLC3], datetime_index)
        df_streaming.at[datetime_index, EMA21_column_name] = np.float64(
            round(ema21_value or 0, 2)
        )

        rsi_value = rsi14.update(candle["close"], datetime_index)
        df_streaming.at[datetime_index, RSI_column_name] = np.float64(
            round(rsi_value or 0, 2)
        )

        # Only the first period super trend value is required to get the swing levels.
        st_fp_direction, st_fp_value = st_fp.update(candle, datetime_index)
        df_streaming.at[datetime_index, supertrend_first_period_value] = np.float64(
            round(st_fp_value or 0, 2)
        )
        df_streaming.at[datetime_index, supertrend_first_period_direction] = (
            st_fp_direction
        )

        st_sp_direction, st_sp_value = st_sp.update(candle, datetime_index)
        df_streaming.at[datetime_index, supertrend_second_period_value] = np.float64(
            round(st_sp_value or 0, 2)
        )
        df_streaming.at[datetime_index, supertrend_second_period_direction] = (
            st_sp_direction
        )

        if rsi_value is not None:
            smoothed_rsi_value = smoothed_rsi.update(rsi_value, datetime_index)
            df_streaming.at[datetime_index, Smoothed_RSI_column_name] = np.float64(
                round(smoothed_rsi_value or 0, 2)
            )

        roc_hlc3_value = roc_hlc3.update(candle[HLC3])
        df_streaming.at[datetime_index, roc_hlc3_column_name] = np.float64(
            round(roc_hlc3_value or 0, 2)
        )

        if ema9_value is not None:
            roc_ema9_value = roc_ema9.update(ema9_value)
            df_streaming.at[datetime_index, roc_ema9_column_name] = np.float64(
                round(roc_ema9_value or 0, 2)
            )
        if ema21_value is not None:
            roc_ema21_value = roc_ema21.update(ema21_value)
            df_streaming.at[datetime_index, roc_ema21_column_name] = np.float64(
                round(roc_ema21_value or 0, 2)
            )
        if rsi_value is not None:
            roc_rsi_value = roc_rsi.update(rsi_value)
            df_streaming.at[datetime_index, roc_rsi_column_name] = np.float64(
                round(roc_rsi_value or 0, 2)
            )

        if smoothed_rsi_value is not None:
            roc_smoothedrsi_value = roc_smoothedrsi.update(smoothed_rsi_value)
            df_streaming.at[datetime_index, roc_smoothed_rsi_column_name] = np.float64(
                round(roc_smoothedrsi_value or 0, 2)
            )
        if ema9_value is not None:
            ppo_hlc3_value = ppo_hlc3.update(candle[HLC3], ema9_value)
            df_streaming.at[datetime_index, ppo_hlc3_ema9_column_name] = np.float64(
                round(ppo_hlc3_value or 0, 2)
            )

        if ema9_value is not None and ema21_value is not None:
            ppo_ema9_value = ppo_ema9.update(ema9_value, ema21_value)
            df_streaming.at[datetime_index, ppo_ema9_ema21_column_name] = np.float64(
                round(ppo_ema9_value or 0, 2)
            )

        if smoothed_rsi_value is not None:
            ppo_rsi_value = ppo_rsi.update(rsi_value, smoothed_rsi_value)
            df_streaming.at[datetime_index, ppo_rsi_emarsi_column_name] = np.float64(
                round(ppo_rsi_value or 0, 2)
            )

    # # Save the indicators into a pickle file in the appropriate folder
    # ema9.save_to_pickle_folder(folder_path)  # Period is 9
    # ema21.save_to_pickle_folder(folder_path)  # Period is 21
    # rsi14.save_to_pickle_folder(folder_path)
    # st_fp.save_to_pickle_folder(folder_path, "fp")
    # st_sp.save_to_pickle_folder(folder_path, "sp")
    # smoothed_rsi.save_to_pickle_folder(folder_path)  # Period is 14
    # # We need to also add the pickle file for PPO and ROC for all indicators here.
    # roc_hlc3.save_to_pickle_folder(folder_path, "hlc3")
    # roc_ema9.save_to_pickle_folder(folder_path, "ema9")
    # roc_ema21.save_to_pickle_folder(folder_path, "ema21")
    # roc_rsi.save_to_pickle_folder(folder_path, "rsi")
    # roc_smoothedrsi.save_to_pickle_folder(folder_path, "smoothedrsi")

    df_streaming = df_streaming.replace(0.0, np.nan).dropna()
    return df_streaming


def loop_market_data(market_data, trading_date: datetime = None):

    year = trading_date.year
    month = trading_date.month
    day = trading_date.day

    for stock_id, stock in market_data.items():
        # We do not have intraday data for this stock hence need to comment
        if stock["symbol"] == "VBL" or stock["symbol"] == "IDFC":
            continue

        folder_path_15min = f"historical/{year}/{month}/{day}/15/{stock_id}/"
        folder_path_5min = f"historical/{year}/{month}/{day}/5/{stock_id}/"
        folder_path_3min = f"historical/{year}/{month}/{day}/3/{stock_id}/"

        df_15min = stock[15]["ohlc"]
        # Need to serialize and save this the current days high and low is the last df entry.
        df_days_high_level, df_days_low_level = calculate_daily_levels(df_15min)
        save_to_csv(df_days_high_level, "high.csv", folder_path_15min)
        save_to_csv(df_days_low_level, "low.csv", folder_path_15min)

        # This complete streaming file is to verify that streaming indicators are working as expected.
        df_15min_streaming = save_streaming_indicators(
            df_15min, stock_id, 15, trading_date=trading_date
        )
        save_to_csv(df_15min_streaming, "indicators.csv", folder_path_15min)

        filter_columns = [
            "high",
            "low",
            supertrend_first_period_value,
            supertrend_first_period_direction,
        ]

        df_15min_supertrend = df_15min_streaming[filter_columns]

        # Need to serialize the swing levels
        swing_levels_15min = SwingLevel()
        # When we are doing streaming the we would have to call the rows method and concat.
        df_15min_swing_levels = swing_levels_15min.loop_data_add_swing_levels(
            df_15min_supertrend
        )
        save_to_csv(df_15min_swing_levels, "swing_levels.csv", folder_path_15min)

        df_5min = stock[5]["ohlc"]
        df_5min_streaming = save_streaming_indicators(
            df_5min, stock_id, 5, trading_date=trading_date
        )
        save_to_csv(df_5min_streaming, "indicators.csv", folder_path_5min)

        df_5min_supertrend = df_5min_streaming[filter_columns]
        swing_levels_5min = SwingLevel()
        df_5min_swing_levels = swing_levels_5min.loop_data_add_swing_levels(
            df_5min_supertrend
        )
        save_to_csv(df_5min_swing_levels, "swing_levels.csv", folder_path_5min)

        df_3min = stock[3]["ohlc"]
        df_3min_streaming = save_streaming_indicators(
            df_3min, stock_id, 3, trading_date=trading_date
        )
        save_to_csv(df_3min_streaming, "indicators.csv", folder_path_3min)

        df_3min_supertrend = df_3min_streaming[filter_columns]
        swing_levels_3min = SwingLevel()
        # We are currently sending the complete data frame. At 3, 5, 15 min interval we would be sending live data
        df_3min_swing_levels = swing_levels_3min.loop_data_add_swing_levels(
            df_3min_supertrend
        )
        save_to_csv(df_3min_swing_levels, "swing_levels.csv", folder_path_3min)


def main():

    client = connect_to_mongodb()
    db = client["nse_stock_data"]

    df_trading_days = get_trading_days_data(db, limit=400)
    last_day_of_month_df = get_start_streaming_trading_days(df_trading_days)

    # generate_streaming_start_trading_day(last_day_of_month_df, db)
    # client.close()
    # return
    start_streaming_trading_day = last_day_of_month_df.iloc[0]["date"]

    # generate_daily_time_frame_data(db, df_trading_days, start_streaming_trading_day)
    # client.close()
    # return

    with timer("retrieve_market_data"):
        # After we have saved the stock in 3, 5, 15 min timeframe in csv. Load it to the market data.
        market_data = retrieve_market_data(start_streaming_trading_day)
        # This takes 1.7 seconds

    with timer("loop_market_data"):
        # Loop through the market data and verify that the streaming indicators and super trend levels work as expected.
        loop_market_data(market_data, start_streaming_trading_day)
        # This takes 112 seconds

    client.close()


if __name__ == "__main__":
    main()
