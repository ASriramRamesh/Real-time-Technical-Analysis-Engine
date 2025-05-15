from ta.pandas_ta import (
    rsi as ta_rsi,
    supertrend as ta_supertrend,
    ema as ta_ema,
    sma as ta_sma,
)
from ta.streaming_indicators import (
    RSI as streaming_rsi,
    SuperTrend as streaming_supertrend,
    EMA as streaming_ema,
)

import pandas as pd
import numpy as np
from pymongo import MongoClient
import pytz
import sqlite3

# Timezone for conversion
kolkata_tz = pytz.timezone("Asia/Kolkata")

# Column names for indicators
HLC3 = "hlc3"
supertrend_first_period_value = "ST_First_Period_Value"
supertrend_first_period_direction = "ST_First_Period_Direction"
supertrend_second_period_value = "ST_Second_Period_Value"
supertrend_second_period_direction = "ST_Second_Period_Direction"
HLC3_column_name = "HLC3"
EMA9_column_name = "EMA9"
EMA21_column_name = "EMA21"
RSI_column_name = "RSI"
Smoothed_RSI_column_name = "SmoothedRSI"

# 2.
# 3. We have to discard 7 days data. Then do the same for 5 min, 3 min.
# 4. For all days Get the supertrend for both periods. RSI and EMA need to be calculated only for the previous day
# 5. Now stream the latest day data and get supertrend, ema, and rsi and compare this with pandas ta.
# 6. If everything is fine then we need to figure out about holding the computed values of supertrend, ema, and rsi.
# 7. We will be removing 1 min data for all but the latest day. RSI and EMA data would be for two days.
# 8. We will compute the super trend data and check it with tradingview.


def connect_to_mongodb(host="localhost", port=27017):
    client = MongoClient(f"mongodb://{host}:{port}/")
    return client


# This table is a import from the stock_master.csv file.
def get_stock_master_data(db):
    collection_name = "stock_master"
    stock_master_col = db[collection_name]
    return pd.DataFrame(list(stock_master_col.find()))


# This returns the days for which we have up loaded intraday data.
def get_trading_days_data(db, limit=10):
    collection_name = "trading_days"
    trading_days_col = db[collection_name]
    return pd.DataFrame(list(trading_days_col.find().sort("date", -1).limit(limit)))


# It gets the one minute data from specified date and sets the datetime to the correct time zone.
def get_stock_data(db, stock_symbol, start_date):
    collection_name = f"stock_{stock_symbol}"
    stock_col = db[collection_name]
    query = {"datetime": {"$gt": start_date}}
    df = pd.DataFrame(list(stock_col.find(query).sort("datetime", 1)))

    # Convert datetime to Asia/Kolkata timezone
    df["datetime"] = df["datetime"].dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
    df.set_index("datetime", inplace=True)

    return df


# resample and aggregate the data based on the time period
def resample_stock_data(df, period="15min", start_date=None):

    if start_date is not None:
        df = df[df.index >= start_date]

    # Resample to specified interval
    df_resampled = df.resample(period).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )

    return df_resampled.dropna()


# For different time frame we have different multiplier the output column is a combination of length and multiplier
def calculate_supertrend(df, value_column, direction_column, length=10, multiplier=2.5):
    df_supertrend = ta_supertrend(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=length,
        multiplier=multiplier,
    )
    value_column_name = f"SUPERT_{length}_{multiplier}"
    direction_column_name = f"SUPERTd_{length}_{multiplier}"

    df[value_column] = df_supertrend[value_column_name].round(2)
    df[direction_column] = df_supertrend[direction_column_name].round(2)


def calculate_rsi(df):
    return None


def main():
    client = connect_to_mongodb()
    db = client["nse_stock_data"]

    df_stock_master = get_stock_master_data(db)
    df_trading_days = get_trading_days_data(db)

    start_date_15min = df_trading_days.iloc[-1]["date"]
    start_date_5min = df_trading_days.iloc[2]["date"]
    start_date_3min = df_trading_days.iloc[1]["date"]

    stock_symbol = df_stock_master.iloc[0]["stock_symbol"]

    df_nse_nifty = get_stock_data(db, stock_symbol, start_date_15min)
    df_nse_nifty_15min = resample_stock_data(df_nse_nifty)

    # For most calculation take the average of high, low and close.
    df_nse_nifty_15min[HLC3] = (
        (
            df_nse_nifty_15min["high"]
            + df_nse_nifty_15min["low"]
            + df_nse_nifty_15min["close"]
        )
        / 3
    ).round(2)

    length = 10
    multiplier = 1.5
    # Calculate Supertrend for the first period and add to the dataframe
    calculate_supertrend(
        df_nse_nifty_15min,
        supertrend_first_period_value,
        supertrend_first_period_direction,
        length=length,
        multiplier=multiplier,
    )

    length = 10
    multiplier = 2.5
    # Calculate Supertrend for the second period and add to the dataframe
    calculate_supertrend(
        df_nse_nifty_15min,
        supertrend_second_period_value,
        supertrend_second_period_direction,
        length=length,
        multiplier=multiplier,
    )

    # Calculate EMA with a period of 9
    df_nse_nifty_15min[EMA9_column_name] = ta_ema(
        df_nse_nifty_15min[HLC3], length=9
    ).round(2)
    # Calculate EMA with a period of 21
    df_nse_nifty_15min[EMA21_column_name] = ta_ema(
        df_nse_nifty_15min[HLC3], length=21
    ).round(2)

    df_nse_nifty_15min[RSI_column_name] = ta_rsi(
        df_nse_nifty_15min["close"], length=14
    ).round(2)

    df_nse_nifty_15min[Smoothed_RSI_column_name] = ta_ema(
        df_nse_nifty_15min[RSI_column_name], length=14
    ).round(2)

    # Uncomment the following lines if you want to save the results to CSV
    # save_to_csv(df_nse_nifty_15min_supertrend, f"{stock_symbol}_supertrend.csv")
    # save_to_csv(df_nse_nifty_15min, f"{stock_symbol}.csv")

    client.close()

    # Filter data for last 2 days
    unique_dates = np.unique(df_nse_nifty_15min.index.date)
    second_last_date = unique_dates[-3]
    df_last_3_days = df_nse_nifty_15min[
        df_nse_nifty_15min.index.date >= second_last_date
    ]

    df_result = df_last_3_days[["open", "high", "low", "close", "hlc3"]].copy()

    streaming_ema9 = streaming_ema(9)
    streaming_ema9_list = []
    streaming_ema21 = streaming_ema(21)
    streaming_ema21_list = []
    streaming_rsi14 = streaming_rsi(14)
    streaming_rsi14_list = []
    ST_10_15 = streaming_supertrend(10, 1.5)
    ST_10_15_VALUE_list = []
    ST_10_15_DIRECTION_list = []
    ST_10_25 = streaming_supertrend(10, 2.5)
    ST_10_25_VALUE_list = []
    ST_10_25_DIRECTION_list = []
    streaming_rsi_ema = streaming_ema(14)
    streaming_rsi_ema_list = []

    for _, candle in df_result.iterrows():
        ema = streaming_ema9.update(candle["hlc3"])
        streaming_ema9_list.append(ema)
        ema = streaming_ema21.update(candle["hlc3"])
        streaming_ema21_list.append(ema)
        rsi = streaming_rsi14.update(candle["close"])
        streaming_rsi14_list.append(rsi)
        st_direction, st_value = ST_10_15.update(candle)
        ST_10_15_VALUE_list.append(st_value)
        ST_10_15_DIRECTION_list.append(st_direction)
        st_direction, st_value = ST_10_25.update(candle)
        ST_10_25_VALUE_list.append(st_value)
        ST_10_25_DIRECTION_list.append(st_direction)
        if rsi is not None:
            rsi_ema = streaming_rsi_ema.update(rsi)
        else:
            rsi_ema = None
        streaming_rsi_ema_list.append(rsi_ema)

    # print(len(ST_10_15_VALUE_list))
    # print(len(df_result))

    df_result[supertrend_first_period_value] = ST_10_15_VALUE_list
    df_result[supertrend_first_period_value] = df_result[
        supertrend_first_period_value
    ].round(2)
    df_result[supertrend_first_period_direction] = ST_10_15_DIRECTION_list
    df_result[supertrend_first_period_direction] = df_result[
        supertrend_first_period_direction
    ].round(2)
    df_result[supertrend_second_period_value] = ST_10_25_VALUE_list
    df_result[supertrend_second_period_value] = df_result[
        supertrend_second_period_value
    ].round(2)
    df_result[supertrend_second_period_direction] = ST_10_25_DIRECTION_list
    df_result[supertrend_second_period_direction] = df_result[
        supertrend_second_period_direction
    ].round(2)

    df_result[EMA9_column_name] = streaming_ema9_list
    df_result[EMA9_column_name] = df_result[EMA9_column_name].round(2)
    df_result[EMA21_column_name] = streaming_ema21_list
    df_result[EMA21_column_name] = df_result[EMA21_column_name].round(2)
    df_result[RSI_column_name] = streaming_rsi14_list
    df_result[RSI_column_name] = df_result[RSI_column_name].round(2)
    df_result[Smoothed_RSI_column_name] = streaming_rsi_ema_list
    df_result[Smoothed_RSI_column_name] = df_result[Smoothed_RSI_column_name].round(2)
    save_to_csv(df_result, f"{stock_symbol}_streaming.csv")


# Uncomment this function if you want to save results to CSV
def save_to_csv(df, filename):
    folder_path = "csv/"
    df.to_csv(folder_path + filename, index=True)


if __name__ == "__main__":
    main()

# ... If we want to filter the data for market hours ...
# def is_market_hours(x):
#     return (x.time() >= pd.Timestamp('09:15').time()) and (x.time() <= pd.Timestamp('15:30').time())

# # Filter for market hours
# df = df[df.index.map(is_market_hours)]

# # Force date time to take the local time.
# CREATE TABLE whatever(
#      ....
#      timestamp DATE DEFAULT (datetime('now','localtime')),
#      ...
# );

# import pandas as pd

# # Sample data with only time values
# data = ['08:30:00', '10:15:00', '13:45:00']

# # Convert the time strings to datetime objects
# time_series = pd.to_datetime(data, format='%H:%M:%S')

# # Create a DataFrame with the time series as index
# df = pd.DataFrame({'value': [1, 2, 3]}, index=time_series)

# print(df)

# import pandas as pd
# import numpy as np

# # Create a NumPy array
# my_array = np.arange(1, 375)

# # Create a DataFrame
# df = pd.DataFrame({'A': [4, 5, 6]})

# # Add the NumPy array as a new column
# df["B"] = my_array

# print(df)
