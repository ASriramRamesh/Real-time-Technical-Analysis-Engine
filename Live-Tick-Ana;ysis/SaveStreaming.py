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
import pickle
from datetime import datetime, timedelta
import os

SuperTrend_First_Period = "fp"
SuperTrend_Second_Period = "sp"


def save_to_csv(df, filename, folder_path="streaming/ohlc/"):
    df.to_csv(folder_path + filename, index=True)


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


def load_dict_from_file(filename):
    with open(filename, "rb") as file:
        loaded_dict = pickle.load(file)
    # print(f"Dictionary loaded from {filename}")
    # # Get the size of the loaded object in bytes
    # size_in_bytes = sys.getsizeof(market_data)
    # print(size_in_bytes)

    return loaded_dict


def retrieve_market_data(current_date: datetime = None):
    market_data = load_dict_from_file("market_data.pkl")

    for stock_id, stock in market_data.items():

        # We do not have intraday data for this stock hence need to comment
        if stock["symbol"] == "VBL" or stock["symbol"] == "IDFC":
            continue

        # # We are first only testing if Nifty 50 levels are getting rendered correctly
        folder_path_15min = "ohlc_csv/15/ohlc/"
        folder_path_5min = "ohlc_csv/5/ohlc/"
        folder_path_3min = "ohlc_csv/3/ohlc/"

        file_name = f"{stock_id}.csv"

        df_15min_all = load_csv_to_dataframe(
            folder_path_15min, file_name, is_date_index=True
        )
        df_5min_all = load_csv_to_dataframe(
            folder_path_5min, file_name, is_date_index=True
        )
        df_3min_all = load_csv_to_dataframe(
            folder_path_3min, file_name, is_date_index=True
        )

        # If we are passing datetime then the latest trading days data is updated to the pickle file.
        if current_date is not None:
            current_date_start = current_date.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            current_date_end = current_date.replace(
                hour=23, minute=59, second=59, microsecond=999999
            )

            df_15min_all = df_15min_all[
                (df_15min_all.index >= current_date_start)
                & (df_15min_all.index <= current_date_end)
            ].copy()

            df_5min_all = df_5min_all[
                (df_5min_all.index >= current_date_start)
                & (df_5min_all.index <= current_date_end)
            ].copy()

            df_3min_all = df_3min_all[
                (df_3min_all.index >= current_date_start)
                & (df_3min_all.index <= current_date_end)
            ].copy()

        df_15min = df_15min_all[["open", "high", "low", "close", "hlc3"]].copy()
        df_5min = df_5min_all[["open", "high", "low", "close", "hlc3"]].copy()
        df_3min = df_3min_all[["open", "high", "low", "close", "hlc3"]].copy()

        stock[15]["ohlc"] = df_15min
        stock[5]["ohlc"] = df_5min
        stock[3]["ohlc"] = df_3min

    return market_data


def load_streaming_indicators(stock_id, time_frame):

    indicators = {
        "ema_9": {"last_update": None, "instance": None},
        "ema_14": {"last_update": None, "instance": None},
        "ema_21": {"last_update": None, "instance": None},
        "rsi_14": {"last_update": None, "instance": None},
        "supertrend_fp": {"last_update": None, "instance": None},
        "supertrend_sp": {"last_update": None, "instance": None},
    }
    time_frame_str = str(time_frame)
    stock_id_str = str(stock_id)
    folder_path = os.path.join(os.getcwd(), "streaming", time_frame_str, stock_id_str)

    for prefix, indicator in indicators.items():
        # Generator expression that returns the first file that has match with prefix
        file_name = next(
            (file for file in os.listdir(folder_path) if file.startswith(prefix)), None
        )

        if file_name:
            # Remove the .pkl extension
            file_name_without_extension = file_name.split(".")[0]
            # We have a standard naming convention -> indicator_period_Date_Time.pkl
            parts = file_name_without_extension.split("_")
            timestamp = f"{parts[2]}_{parts[3]}"
            timestamp_date = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
            indicator["last_update"] = timestamp_date

            # Get the instance of EMA 9 period
            if prefix == "ema_9":
                indicator["instance"] = streaming_ema.load_from_pickle(
                    time_frame_str, stock_id_str, str(9), timestamp
                )
            # Get the instance of EMA of RSI for 14 period
            elif prefix == "ema_14":
                indicator["instance"] = streaming_ema.load_from_pickle(
                    time_frame_str, stock_id_str, str(14), timestamp
                )
            # Get the instance of EMA 21 period
            elif prefix == "ema_21":
                indicator["instance"] = streaming_ema.load_from_pickle(
                    time_frame_str, stock_id_str, str(21), timestamp
                )
            # Get the instance of RSI 14 period
            elif prefix == "rsi_14":
                indicator["instance"] = streaming_rsi.load_from_pickle(
                    time_frame_str, stock_id_str, str(14), timestamp
                )
            # Get the instance of SuperTrend first period
            elif prefix == "supertrend_fp":
                indicator["instance"] = streaming_supertrend.load_from_pickle(
                    time_frame_str, stock_id_str, SuperTrend_First_Period, timestamp
                )
            # Get the instance of SuperTrend second period
            elif prefix == "supertrend_sp":
                indicator["instance"] = streaming_supertrend.load_from_pickle(
                    time_frame_str, stock_id_str, SuperTrend_Second_Period, timestamp
                )
            # Get the filepath to delete the file
            file_path = os.path.join(folder_path, file_name)
            try:
                # Delete the file
                os.remove(file_path)
            except OSError as e:
                print(f"Error deleting file {file_name}: {e}")
    # Return the instance of all the indicators
    return indicators


def save_streaming_indicators(df, stock_id, time_frame, load_modify_overwrite=False):

    supertrend_period_length = 10

    if time_frame == 3:
        supertrend_first_period_multiplier = 2
        supertrend_second_period_multiplier = 3
    elif time_frame == 5:
        supertrend_first_period_multiplier = 2
        supertrend_second_period_multiplier = 3
    else:
        supertrend_first_period_multiplier = 1.5
        supertrend_second_period_multiplier = 2.5

    ema9 = None
    ema21 = None
    rsi14 = None
    st_fp = None
    st_sp = None
    rsi_ema = None

    if load_modify_overwrite:
        # load existing pickle file for the indicators
        indicators = load_streaming_indicators(stock_id, time_frame)
        ema9 = indicators["ema_9"]["instance"]
        ema21 = indicators["ema_21"]["instance"]
        rsi14 = indicators["rsi_14"]["instance"]
        st_fp = indicators["supertrend_fp"]["instance"]
        st_sp = indicators["supertrend_sp"]["instance"]
        rsi_ema = indicators["ema_14"]["instance"]

    # If the pickle file is not found then we assume that we would have to generate from scratch.
    if ema9 is None:
        ema9 = streaming_ema(9)
    if ema21 is None:
        ema21 = streaming_ema(21)
    if rsi14 is None:
        rsi14 = streaming_rsi(14)
    if st_fp is None:
        st_fp = streaming_supertrend(
            supertrend_period_length, supertrend_first_period_multiplier
        )
    if st_sp is None:
        st_sp = streaming_supertrend(
            supertrend_period_length, supertrend_second_period_multiplier
        )
    if rsi_ema is None:
        rsi_ema = streaming_ema(14)

    # We keep updating the values for each row of OHLC
    for datetime_index, candle in df.iterrows():
        # We have to update the datetime also
        ema9.update(candle["hlc3"], datetime_index)
        ema21.update(candle["hlc3"], datetime_index)
        rsi = rsi14.update(candle["close"], datetime_index)
        st_fp.update(candle, datetime_index)
        st_sp.update(candle, datetime_index)
        if rsi is not None:
            rsi_ema.update(rsi, datetime_index)

    time_frame_str = str(time_frame)
    stock_id_str = str(stock_id)

    # Save the indicators into a pickle file in the appropriate folder
    ema9.save_to_pickle(time_frame_str, stock_id_str)  # Period is 9
    ema21.save_to_pickle(time_frame_str, stock_id_str)  # Period is 21
    rsi14.save_to_pickle(time_frame_str, stock_id_str)
    st_fp.save_to_pickle(time_frame_str, stock_id_str, SuperTrend_First_Period)
    st_sp.save_to_pickle(time_frame_str, stock_id_str, SuperTrend_Second_Period)
    rsi_ema.save_to_pickle(time_frame_str, stock_id_str)  # Period is 14


# We loop through the market data and keep updating the streaming indicators for the stock and time frame.
def loop_market_data(market_data, load_modify_overwrite=False):

    for stock_id, stock in market_data.items():
        # We do not have intraday data for this stock hence need to comment
        if stock["symbol"] == "VBL" or stock["symbol"] == "IDFC":
            continue

        df_15min = stock[15]["ohlc"]
        save_streaming_indicators(
            df_15min, stock_id, 15, load_modify_overwrite=load_modify_overwrite
        )

        df_5min = stock[5]["ohlc"]
        save_streaming_indicators(
            df_5min, stock_id, 5, load_modify_overwrite=load_modify_overwrite
        )

        df_3min = stock[3]["ohlc"]
        save_streaming_indicators(
            df_3min, stock_id, 3, load_modify_overwrite=load_modify_overwrite
        )

    # Filter data for last 2 days
    # unique_dates = np.unique(df_nse_nifty_15min.index.date)
    # second_last_date = unique_dates[-3]
    # df_last_3_days = df_nse_nifty_15min[
    #     df_nse_nifty_15min.index.date >= second_last_date
    # ]

    # df_result = df_last_3_days[["open", "high", "low", "close", "hlc3"]].copy()

    # # print(len(ST_10_15_VALUE_list))
    # # print(len(df_result))

    # df_result[supertrend_first_period_value] = ST_10_15_VALUE_list
    # df_result[supertrend_first_period_value] = df_result[
    #     supertrend_first_period_value
    # ].round(2)
    # df_result[supertrend_first_period_direction] = ST_10_15_DIRECTION_list
    # df_result[supertrend_first_period_direction] = df_result[
    #     supertrend_first_period_direction
    # ].round(2)
    # df_result[supertrend_second_period_value] = ST_10_25_VALUE_list
    # df_result[supertrend_second_period_value] = df_result[
    #     supertrend_second_period_value
    # ].round(2)
    # df_result[supertrend_second_period_direction] = ST_10_25_DIRECTION_list
    # df_result[supertrend_second_period_direction] = df_result[
    #     supertrend_second_period_direction
    # ].round(2)

    # df_result[EMA9_column_name] = streaming_ema9_list
    # df_result[EMA9_column_name] = df_result[EMA9_column_name].round(2)
    # df_result[EMA21_column_name] = streaming_ema21_list
    # df_result[EMA21_column_name] = df_result[EMA21_column_name].round(2)
    # df_result[RSI_column_name] = streaming_rsi14_list
    # df_result[RSI_column_name] = df_result[RSI_column_name].round(2)
    # df_result[Smoothed_RSI_column_name] = streaming_rsi_ema_list
    # df_result[Smoothed_RSI_column_name] = df_result[Smoothed_RSI_column_name].round(2)
    # save_to_csv(df_result, f"{stock_symbol}_streaming.csv")


def main():
    # We run streaming indicators on all the days in the csv file
    # market_data = retrieve_market_data()

    # If we want to update the current date values
    current_date = datetime.now(pd.Timestamp("now", tz="UTC+05:30").tz)

    # If we want to update the previous days value
    previous_date = current_date - timedelta(days=1)

    # We filter data of current day to update the streaming indicators pickle file
    # market_data = retrieve_market_data(current_date)
    # We filter data of previous day to update the streaming indicators pickle file
    market_data = retrieve_market_data(previous_date)

    # When we want to build the pickle files from scratch
    # loop_market_data(market_data)
    # When we want to only update the current date ohlc computation to the pickle file
    loop_market_data(market_data, load_modify_overwrite=True)


if __name__ == "__main__":
    main()
