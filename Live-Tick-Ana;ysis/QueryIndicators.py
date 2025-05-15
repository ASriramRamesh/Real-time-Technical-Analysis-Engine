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
from typing import Dict
import pytz
from datetime import datetime
import sqlite3
import pickle
import os
import sys

# Timezone for conversion
kolkata_tz = pytz.timezone("Asia/Kolkata")

# Column names for indicators
HLC3 = "hlc3"
supertrend_first_period_value = "ST_First_Period_Value"
supertrend_first_period_direction = "ST_First_Period_Direction"
supertrend_second_period_value = "ST_Second_Period_Value"
supertrend_second_period_direction = "ST_Second_Period_Direction"
EMA9_column_name = "EMA9"
EMA21_column_name = "EMA21"
ppo_hlc3_ema9_column_name = "PPO_HLC3_EMA9"

ema50_column_name = "ema_50"
ppo_hlc3ema21_column_name = "ppo_hlc3ema21"
ppo_hlc3st_fp_column_name = "ppo_hlc3st_fp"
ppo_hlc3st_sp_column_name = "ppo_hlc3st_sp"
ind_slope_column_name = "ind_slope"
ind_y_intercept_column_name = "ind_y-intercept"
ind_r_squared_column_name = "ind_r-squared"
ind_p_coef_column_name = "ind_p-coef"
dep_slope_column_name = "dep_slope"
dep_y_intercept_column_name = "dep_y-intercept"
dep_r_squared_column_name = "dep_r-squared"
dep_p_coef_column_name = "dep_p-coef"

Nifty_50 = 1
Nifty_Bank = 2
Nifty_IT = 3
Nifty_AUTO = 4
Nifty_ENERGY = 5
Nifty_INFRA = 6
Nifty_FMCG = 7
Nifty_METAL = 8
Nifty_PHARMA = 9
Nifty_PSU_Bank = 10
Nifty_Private_Bank = 11
Nifty_Fin_Service = 12


def connect_to_mongodb(host="localhost", port=27017):
    client = MongoClient(f"mongodb://{host}:{port}/")
    return client


# This table is a import from the stock_master.csv file.
def get_stock_master_data(db):
    collection_name = "stock_master"
    stock_master_col = db[collection_name]
    df = pd.DataFrame(list(stock_master_col.find()))
    df_sorted = df.sort_values(by="stock_id")
    return df_sorted


# This returns the days for which we have up loaded intraday data.
def get_trading_days_data(db, limit=10, query=None):
    collection_name = "trading_days"
    trading_days_col = db[collection_name]
    if query is None:
        return pd.DataFrame(list(trading_days_col.find().sort("date", -1).limit(limit)))
    else:
        return pd.DataFrame(
            list(trading_days_col.find(query).sort("date", 1).limit(limit))
        )


# It gets the one minute data from specified date and sets the datetime to the correct time zone.
def get_stock_data(db, stock_symbol, start_date, end_date=None):
    collection_name = f"stock_{stock_symbol}"
    stock_col = db[collection_name]
    if end_date is None:
        query = {"datetime": {"$gt": start_date}}
    else:
        query = {"datetime": {"$gte": start_date, "$lte": end_date}}
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


# Set the super trend column names and values for different time frame, period and multiplier
def set_supertrendvalues_dataframe(df, period_length, period_multiplier, isFirst=False):
    df_supertrend = ta_supertrend(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        length=period_length,
        multiplier=period_multiplier,
    )

    value_column_name = f"SUPERT_{period_length}_{period_multiplier:.1f}"
    direction_column_name = f"SUPERTd_{period_length}_{period_multiplier:.1f}"

    supertrend_column_value = (
        supertrend_first_period_value if isFirst else supertrend_second_period_value
    )
    supertrend_column_direction = (
        supertrend_first_period_direction
        if isFirst
        else supertrend_second_period_direction
    )

    df[supertrend_column_value] = df_supertrend[value_column_name].round(2)
    df[supertrend_column_direction] = df_supertrend[direction_column_name].round(2)


# For different dataframes calculate super trend.
def calculate_supertrend(df_15, df_5, df_3):
    first_period_15_min_length = 10
    first_period_15_min_multiplier = 1.5
    second_period_15_min_length = 10
    second_period_15_min_multiplier = 2.5
    first_period_5_min_length = 10
    first_period_5_min_multiplier = 2
    second_period_5_min_length = 10
    second_period_5_min_multiplier = 3
    first_period_3_min_length = 10
    first_period_3_min_multiplier = 3
    second_period_3_min_length = 10
    second_period_3_min_multiplier = 4

    # 15 min first period super trend
    set_supertrendvalues_dataframe(
        df_15, first_period_15_min_length, first_period_15_min_multiplier, isFirst=True
    )

    # 15 min first period super trend
    set_supertrendvalues_dataframe(
        df_15, second_period_15_min_length, second_period_15_min_multiplier
    )

    # 5 min first period super trend
    set_supertrendvalues_dataframe(
        df_5, first_period_5_min_length, first_period_5_min_multiplier, isFirst=True
    )

    # 5 min first period super trend
    set_supertrendvalues_dataframe(
        df_5, second_period_5_min_length, second_period_5_min_multiplier
    )

    # 3 min first period super trend
    set_supertrendvalues_dataframe(
        df_3, first_period_3_min_length, first_period_3_min_multiplier, isFirst=True
    )

    # 3 min first period super trend
    set_supertrendvalues_dataframe(
        df_3, second_period_3_min_length, second_period_3_min_multiplier
    )


def calculate_ema(df_15, df_5, df_3):
    # Calculate EMA with a period of 9 and 15 min dataframe
    df_15[EMA9_column_name] = ta_ema(df_15[HLC3], length=9).round(2)
    # Calculate EMA with a period of 21 and 15 min dataframe
    df_15[EMA21_column_name] = ta_ema(df_15[HLC3], length=21).round(2)
    # Calculate EMA with a period of 9 and 5 min dataframe
    df_5[EMA9_column_name] = ta_ema(df_5[HLC3], length=9).round(2)
    # Calculate EMA with a period of 21 and 5 min dataframe
    df_5[EMA21_column_name] = ta_ema(df_5[HLC3], length=21).round(2)
    # Calculate EMA with a period of 9 and 3 min dataframe
    df_3[EMA9_column_name] = ta_ema(df_3[HLC3], length=9).round(2)
    # Calculate EMA with a period of 21 and 3 min dataframe
    df_3[EMA21_column_name] = ta_ema(df_3[HLC3], length=21).round(2)


# def calculate_rsi(df_15, df_5, df_3):
#     # Calculate RSI for 15 min dataframe
#     df_15[RSI_column_name] = ta_rsi(df_15["close"], length=14).round(2)
#     # Calculate EMA for RSI for 15 min dataframe
#     df_15[Smoothed_RSI_column_name] = ta_ema(df_15[RSI_column_name], length=14).round(2)
#     # Calculate RSI for 15 min dataframe
#     df_5[RSI_column_name] = ta_rsi(df_5["close"], length=14).round(2)
#     # Calculate EMA for RSI for 15 min dataframe
#     df_5[Smoothed_RSI_column_name] = ta_ema(df_5[RSI_column_name], length=14).round(2)
#     # Calculate RSI for 15 min dataframe
#     df_3[RSI_column_name] = ta_rsi(df_3["close"], length=14).round(2)
#     # Calculate EMA for RSI for 15 min dataframe
#     df_3[Smoothed_RSI_column_name] = ta_ema(df_3[RSI_column_name], length=14).round(2)


# Get the intraday data and then compute the high and low of unique dates
def calculate_daily_levels(df):
    df_high_low = (
        df.resample("D")[["high", "low"]]
        .agg({"high": "max", "low": "min"})
        .reset_index()
    )
    df_high_low = df_high_low.dropna(subset=["high"])

    df_levels = pd.DataFrame(columns=["date", "high", "low"])

    for index, row in df_high_low.iterrows():
        date = row["datetime"]
        high = row["high"]
        low = row["low"]

        # Filter future dates
        future_high_low = df_high_low[df_high_low["datetime"] > date]

        # Check if high falls between future highs and lows
        high_between = (future_high_low["low"] <= high) & (
            high <= future_high_low["high"]
        )
        if high_between.any():
            high_level = 0
        else:
            high_level = high

        # Check if low falls between future highs and lows
        low_between = (future_high_low["low"] <= low) & (low <= future_high_low["high"])
        if low_between.any():
            low_level = 0
        else:
            low_level = low

        # Append to df_levels
        df_levels = pd.concat(
            [
                df_levels if not df_levels.empty else None,
                pd.DataFrame(
                    {"date": [date], "high": [high_level], "low": [low_level]}
                ),
            ],
            ignore_index=True,
        )

    df_levels["days_past"] = len(df_levels) - df_levels.index.values
    df_days_high_level = df_levels.query("high != 0")[
        ["date", "high", "days_past"]
    ].copy()
    df_days_low_level = df_levels.query("low != 0")[["date", "low", "days_past"]].copy()
    return df_days_high_level, df_days_low_level


# Set the high and low values comparing it with the current candle.
def set_period_values(candle, datetime_index, first_period):
    if (
        first_period["high_till_now"] == 0
        or first_period["high_till_now"] < candle["high"]
    ):
        first_period["high_till_now"] = candle["high"]
        first_period["high_till_now_datetime"] = datetime_index

    if (
        first_period["low_till_now"] == 0
        or first_period["low_till_now"] > candle["low"]
    ):
        first_period["low_till_now"] = candle["low"]
        first_period["low_till_now_datetime"] = datetime_index

    first_period["direction"] = candle[supertrend_first_period_direction]
    first_period["value"] = candle[supertrend_first_period_value]

    return first_period


def create_empty_timeseries_dataframes():
    df_start_up_move = pd.DataFrame(
        columns=[
            "datetime",
            "support",
            "resistance",
            "is_up_move",
            "trigger_datetime",
        ]
    )
    df_start_up_move.set_index("datetime", inplace=True)

    df_start_down_move = pd.DataFrame(
        columns=[
            "datetime",
            "support",
            "resistance",
            "is_down_move",
            "trigger_datetime",
        ]
    )
    df_start_down_move.set_index("datetime", inplace=True)

    return df_start_up_move, df_start_down_move


def create_empty_timeseries_dataframes_rows():
    new_start_of_upmove_row = {
        "datetime": 0,
        "support": 0,
        "resistance": 0,
        "is_up_move": 0,
        "trigger_datetime": 0,
    }

    new_start_of_downmove_row = {
        "datetime": 0,
        "support": 0,
        "resistance": 0,
        "is_down_move": 0,
        "trigger_datetime": 0,
    }

    return new_start_of_upmove_row, new_start_of_downmove_row


def create_new_start_of_upmove(
    new_start_of_upmove_row, first_period, trigger_datetime, is_up_move=True
):
    if is_up_move:
        new_start_of_upmove_row["datetime"] = first_period["low_till_now_datetime"]
    else:
        new_start_of_upmove_row["datetime"] = first_period["high_till_now_datetime"]

    new_start_of_upmove_row["support"] = first_period["low_till_now"]
    new_start_of_upmove_row["resistance"] = first_period["high_till_now"]
    new_start_of_upmove_row["is_up_move"] = is_up_move
    new_start_of_upmove_row["trigger_datetime"] = trigger_datetime

    return new_start_of_upmove_row


def create_new_start_of_downmove(
    new_start_of_downmove_row, first_period, trigger_datetime, is_down_move=True
):
    if is_down_move:
        new_start_of_downmove_row["datetime"] = first_period["high_till_now_datetime"]
    else:
        new_start_of_downmove_row["datetime"] = first_period["low_till_now_datetime"]

    new_start_of_downmove_row["support"] = first_period["low_till_now"]
    new_start_of_downmove_row["resistance"] = first_period["high_till_now"]
    new_start_of_downmove_row["is_down_move"] = is_down_move
    new_start_of_downmove_row["trigger_datetime"] = trigger_datetime

    return new_start_of_downmove_row


def create_period_dict():
    period_dict = {
        "high_till_now": 0,
        "high_till_now_datetime": None,
        "low_till_now": 0,
        "low_till_now_datetime": None,
        "direction": 0,
        "value": 0,
        "flat_count": 0,
    }

    return period_dict


def initialize_period_dict(bar_datetime, candle):
    period_dict = {
        "high_till_now": candle["high"],
        "high_till_now_datetime": bar_datetime,
        "low_till_now": candle["low"],
        "low_till_now_datetime": bar_datetime,
        "direction": candle[supertrend_first_period_direction],
        "value": candle[supertrend_first_period_value],
        "flat_count": 0,
    }

    return period_dict


def should_insert_trend_pullback_row(df, is_start_of_up_move=False):
    should_insert = False

    # Since we cover a long duration we ignore swing points till we have a crossover in supertrend.
    if len(df) == 0:
        return should_insert

    last_row = df.iloc[-1]
    if is_start_of_up_move:
        is_up_move = last_row["is_up_move"]
        if is_up_move:
            should_insert = True
    else:
        is_down_move = last_row["is_down_move"]
        if is_down_move:
            should_insert = True

    return should_insert


# This will check if close is higher than resistance of last record and there is a pullback
def should_insert_trend_continuation_row(df, close, is_start_of_up_move=False):
    should_insert = False

    # Since we cover a long duration we ignore swing points till we have a crossover in supertrend.
    if len(df) == 0:
        return should_insert

    last_row = df.iloc[-1]
    if is_start_of_up_move:
        is_up_move = last_row["is_up_move"]
        swing_high = last_row["resistance"]
        if not is_up_move and close > swing_high:
            should_insert = True
    else:
        is_down_move = last_row["is_down_move"]
        swing_low = last_row["support"]
        if not is_down_move and close < swing_low:
            should_insert = True

    return should_insert


def loop_min_data_create_swing_levels(df_time_frame):
    first_period = create_period_dict()

    df_start_up_move, df_start_down_move = create_empty_timeseries_dataframes()

    for datetime_index, candle in df_time_frame.iterrows():

        new_start_of_upmove_row, new_start_of_downmove_row = (
            create_empty_timeseries_dataframes_rows()
        )

        # This is the first candle so assign values to both periods and continue
        if first_period["direction"] == 0:
            first_period = initialize_period_dict(datetime_index, candle)

        # There is two possibilities first is crossover to down trend and second is crossover to up trend
        elif candle[supertrend_first_period_direction] != first_period["direction"]:
            # 1. We have to save to either start of up move or start of down move
            if candle[supertrend_first_period_direction] == 1:
                # We have crossover to an uptrend
                new_start_of_upmove_row = create_new_start_of_upmove(
                    new_start_of_upmove_row,
                    first_period,
                    datetime_index,
                    is_up_move=True,
                )
                new_row = pd.DataFrame([new_start_of_upmove_row]).set_index("datetime")
                df_start_up_move = pd.concat(
                    [df_start_up_move if not df_start_up_move.empty else None, new_row]
                )
            else:
                # We have crossover to an downtrend
                new_start_of_downmove_row = create_new_start_of_downmove(
                    new_start_of_downmove_row,
                    first_period,
                    datetime_index,
                    is_down_move=True,
                )
                new_row = pd.DataFrame([new_start_of_downmove_row]).set_index(
                    "datetime"
                )
                df_start_down_move = pd.concat(
                    [
                        df_start_down_move if not df_start_down_move.empty else None,
                        new_row,
                    ]
                )

            # reset the period to the current candle
            first_period = initialize_period_dict(datetime_index, candle)
        else:
            # This condition is to add pullback when first period super trend flattens
            if candle[supertrend_first_period_value] == first_period["value"]:
                if first_period["flat_count"] == 0:
                    first_period = set_period_values(
                        candle, datetime_index, first_period
                    )
                    first_period["flat_count"] = 1
                elif first_period["flat_count"] == 1:
                    # 1. If this is start of down move in a uptrend then we have to check if there is a prior record of start of down move.
                    # 2. If no prior start of down move We have to save to either start of up move or start of down move
                    if candle[supertrend_first_period_direction] == 1:
                        # This is a pullback in a up move
                        # If the last record has column is_up_move as True then only we insert a record
                        should_insert = should_insert_trend_pullback_row(
                            df_start_up_move, is_start_of_up_move=True
                        )

                        if should_insert:
                            new_start_of_upmove_row = create_new_start_of_upmove(
                                new_start_of_upmove_row,
                                first_period,
                                datetime_index,
                                is_up_move=False,
                            )
                            new_start_of_upmove_row["support"] = candle["low"]
                            new_row = pd.DataFrame([new_start_of_upmove_row]).set_index(
                                "datetime"
                            )
                            df_start_up_move = pd.concat(
                                [
                                    (
                                        df_start_up_move
                                        if not df_start_up_move.empty
                                        else None
                                    ),
                                    new_row,
                                ]
                            )
                    else:
                        # This is a pullback in a down move
                        should_insert = should_insert_trend_pullback_row(
                            df_start_down_move, is_start_of_up_move=False
                        )

                        if should_insert:
                            new_start_of_downmove_row = create_new_start_of_downmove(
                                new_start_of_downmove_row,
                                first_period,
                                datetime_index,
                                is_down_move=False,
                            )
                            new_start_of_downmove_row["resistance"] = candle["high"]
                            new_row = pd.DataFrame(
                                [new_start_of_downmove_row]
                            ).set_index("datetime")
                            df_start_down_move = pd.concat(
                                [
                                    (
                                        df_start_down_move
                                        if not df_start_down_move.empty
                                        else None
                                    ),
                                    new_row,
                                ]
                            )

                    first_period = initialize_period_dict(datetime_index, candle)
                    first_period["flat_count"] = 2
                else:
                    # We have already registered the pullback and now we either have crossover or continuation of trend
                    first_period = set_period_values(
                        candle, datetime_index, first_period
                    )
                    first_period["flat_count"] += 1
            else:
                # This part of code gets executed when price is in trend continuation.
                if candle[supertrend_first_period_direction] == 1:
                    # Since we cover a long duration we ignore swing points till we have a crossover.
                    # Verifying prior record is a pullback
                    should_insert_record = should_insert_trend_continuation_row(
                        df_start_up_move,
                        candle["high"],
                        is_start_of_up_move=True,
                    )
                    if should_insert_record:
                        new_start_of_upmove_row = create_new_start_of_upmove(
                            new_start_of_upmove_row,
                            first_period,
                            datetime_index,
                            is_up_move=True,
                        )
                        new_row = pd.DataFrame([new_start_of_upmove_row]).set_index(
                            "datetime"
                        )
                        df_start_up_move = pd.concat(
                            [
                                (
                                    df_start_up_move
                                    if not df_start_up_move.empty
                                    else None
                                ),
                                new_row,
                            ]
                        )

                        first_period = initialize_period_dict(datetime_index, candle)
                    else:
                        first_period = set_period_values(
                            candle, datetime_index, first_period
                        )
                else:
                    should_insert_record = should_insert_trend_continuation_row(
                        df_start_down_move,
                        candle["low"],
                        is_start_of_up_move=False,
                    )
                    if should_insert_record:
                        new_start_of_downmove_row = create_new_start_of_downmove(
                            new_start_of_downmove_row,
                            first_period,
                            datetime_index,
                            is_down_move=True,
                        )
                        new_row = pd.DataFrame([new_start_of_downmove_row]).set_index(
                            "datetime"
                        )
                        df_start_down_move = pd.concat(
                            [
                                (
                                    df_start_down_move
                                    if not df_start_down_move.empty
                                    else None
                                ),
                                new_row,
                            ]
                        )

                        first_period = initialize_period_dict(datetime_index, candle)
                    else:
                        first_period = set_period_values(
                            candle, datetime_index, first_period
                        )
                first_period["flat_count"] = 0

    return df_start_up_move, df_start_down_move


# Get the intraday data and then compute the high and low of unique dates
def calculate_supertrend_levels(df):

    df_start_up_move, df_start_down_move = loop_min_data_create_swing_levels(df)
    df_start_up_move["is_up_trend"] = True
    df_start_down_move["is_up_trend"] = False
    df_start_down_move["is_up_move"] = df_start_down_move["is_down_move"].apply(
        lambda x: not x
    )

    df_start_up_move["level"] = df_start_up_move.apply(
        lambda row: row["support"] if row["is_up_move"] else row["resistance"], axis=1
    )
    df_start_down_move["level"] = df_start_down_move.apply(
        lambda row: row["support"] if row["is_up_move"] else row["resistance"], axis=1
    )

    df_up = df_start_up_move[
        ["is_up_trend", "is_up_move", "level", "trigger_datetime"]
    ].copy()
    df_down = df_start_down_move[
        ["is_up_trend", "is_up_move", "level", "trigger_datetime"]
    ].copy()

    if not df_up.empty:
        df_swing_levels = pd.concat(
            [df_up, df_down if not df_down.empty else None]
        ).sort_index()
    elif not df_down.empty:
        df_swing_levels = pd.concat(
            [df_up if not df_up.empty else None, df_down]
        ).sort_index()
    else:
        df_swing_levels = pd.DataFrame(
            columns=["is_up_trend", "is_up_move", "level", "trigger_datetime"]
        )

    return df_swing_levels


# For most calculation take the average of high, low and close.
def calculate_hlc3(df_15, df_5, df_3):
    df_15[HLC3] = ((df_15["high"] + df_15["low"] + df_15["close"]) / 3).round(2)
    df_5[HLC3] = ((df_5["high"] + df_5["low"] + df_5["close"]) / 3).round(2)
    df_3[HLC3] = ((df_3["high"] + df_3["low"] + df_3["close"]) / 3).round(2)


# Set the ppo for the fast and slow column
def calculate_set_ppo_columns(df, ppo_column_name, fast_column_name, slow_column_name):
    df[ppo_column_name] = ta_ppo(
        fast=df[fast_column_name], slow=df[slow_column_name]
    ).round(2)


# def calculate_ppo_timeframes(df):
#     calculate_set_ppo_columns(df, ppo_hlc3_ema9_column_name, HLC3, EMA9_column_name)
#     calculate_set_ppo_columns(
#         df, ppo_ema9_ema21_column_name, EMA9_column_name, EMA21_column_name
#     )
#     calculate_set_ppo_columns(
#         df, ppo_rsi_emarsi_column_name, RSI_column_name, Smoothed_RSI_column_name
#     )


# For all time frames create new columns to compare
# hlc3 and ema9 -> PPO_HLC3_EMA9
# ema9 and ema21 -> PPO_EMA9_EMA21
# rsi and smoothed ma of rsi -> PPO_RSI_EMARSI
# def calculate_ppo(df_15, df_5, df_3):
#     # Calculate PPO for hlc3 15 min dataframe
#     calculate_ppo_timeframes(df_15)
#     calculate_ppo_timeframes(df_5)
#     calculate_ppo_timeframes(df_3)


# def calculate_roc_timeframes(df):
#     df[roc_hlc3_column_name] = ta_roc(df[HLC3]).round(2)
#     df[roc_ema9_column_name] = ta_roc(df[EMA9_column_name]).round(2)
#     df[roc_ema21_column_name] = ta_roc(df[EMA21_column_name]).round(2)
#     df[roc_rsi_column_name] = ta_roc(df[RSI_column_name]).round(2)
#     df[roc_smoothed_rsi_column_name] = ta_roc(df[Smoothed_RSI_column_name]).round(2)


# For all time frames we compare the change in current value with the previous value
# hlc3 -> ROC_HLC3
# ema9 -> ROC_EMA9
# ema21 -> ROC_EMA21
# rsi -> ROC_RSI
# smoothed ma of rsi -> ROC_EMARSI
# def calculate_roc(df_15, df_5, df_3):
#     # Calculate PPO for hlc3 15 min dataframe
#     calculate_roc_timeframes(df_15)
#     calculate_roc_timeframes(df_5)
#     calculate_roc_timeframes(df_3)


# Function to save the dictionary to a file
def save_dict_to_file(dictionary, filename):
    with open(filename, "wb") as file:
        pickle.dump(dictionary, file)
    print(f"Dictionary saved to {filename}")


# Function to load the dictionary from a file
def load_dict_from_file(filename):
    with open(filename, "rb") as file:
        loaded_dict = pickle.load(file)
    # print(f"Dictionary loaded from {filename}")
    # # Get the size of the loaded object in bytes
    # size_in_bytes = sys.getsizeof(market_data)
    # print(size_in_bytes)

    return loaded_dict


# This would be loaded with indicator class which would be saved in a pickle file.
def get_indicators_schema():
    return {
        1: None,
        2: None,
        3: None,
        4: None,
        5: None,
        6: None,
        7: None,
    }


def get_stock_schema():
    return {
        "ohlc": pd.DataFrame(columns=["open", "high", "low", "close"]),
        "swing_levels": pd.DataFrame(
            columns=[
                "datetime",
                "is_up_trend",
                "is_up_move",
                "level",
                "trigger_datetime",
            ]
        ),
    }


# This will set the market_data.pkl file
def set_market_data(df, market_data):
    indicator_data: Dict[int, Dict[int, Dict[int, object]]] = {}
    # This is the first loop where we created the structure of market data.
    for _, stock in df.iterrows():
        stock_id = stock["stock_id"]
        stock_symbol = stock["stock_symbol"]
        # We do not have intraday data for this stock hence need to comment
        if stock_symbol == "VBL" or stock_symbol == "IDFC" or stock_symbol == "DRREDDY":
            continue

        dict_15 = get_stock_schema()
        dict_5 = get_stock_schema()
        dict_3 = get_stock_schema()

        dict_stock = {
            "symbol": stock_symbol,
            "sector_ids": [],
            "highs": pd.DataFrame(columns=["date", "high", "days_past"]),
            "lows": pd.DataFrame(columns=["date", "low", "days_past"]),
            "close": 0,
            "closing_range_high": 0,
            "closing_range_low": 0,
            15: dict_15,
            5: dict_5,
            3: dict_3,
        }
        market_data[stock_id] = dict_stock

        event_indicators_15 = get_indicators_schema()
        event_indicators_5 = get_indicators_schema()
        event_indicators_3 = get_indicators_schema()
        dict_indicators = {
            15: event_indicators_15,
            5: event_indicators_5,
            3: event_indicators_3,
        }
        stock_id = int(stock_id)
        indicator_data[stock_id] = dict_indicators

    # Now we map the stocks to sector and sector to index
    for _, stock in df.iterrows():
        stock_id = stock["stock_id"]
        stock_symbol = stock["stock_symbol"]
        # We do not have intraday data for this stock hence need to comment
        if stock_symbol == "VBL" or stock_symbol == "IDFC" or stock_symbol == "DRREDDY":
            continue
        sector_id = stock["sector_id"]
        market_data[sector_id]["sector_ids"].append(stock_id)

    # Adding the Banks to Finnifty
    market_data[Nifty_Fin_Service]["sector_ids"].append(17)  # SBI
    market_data[Nifty_Private_Bank]["sector_ids"].append(13)  # HDFC Bank
    market_data[Nifty_Private_Bank]["sector_ids"].append(14)  # ICICI Bank
    market_data[Nifty_Private_Bank]["sector_ids"].append(15)  # Kotak Bank
    market_data[Nifty_Private_Bank]["sector_ids"].append(16)  # Axis Bank

    # Adding the PSU Banks to PSU sector
    market_data[Nifty_PSU_Bank]["sector_ids"].append(17)  # SBI
    market_data[Nifty_PSU_Bank]["sector_ids"].append(19)  # Bank of Baroda
    market_data[Nifty_PSU_Bank]["sector_ids"].append(22)  # PNB

    # Adding the Private Banks to Private sector
    market_data[Nifty_Private_Bank]["sector_ids"].append(13)  # HDFC Bank
    market_data[Nifty_Private_Bank]["sector_ids"].append(14)  # ICICI Bank
    market_data[Nifty_Private_Bank]["sector_ids"].append(15)  # Kotak Bank
    market_data[Nifty_Private_Bank]["sector_ids"].append(16)  # Axis Bank
    market_data[Nifty_Private_Bank]["sector_ids"].append(18)  # IndusInd Bank
    market_data[Nifty_Private_Bank]["sector_ids"].append(20)  # Federal Bank
    market_data[Nifty_Private_Bank]["sector_ids"].append(23)  # IDFC Bank

    # We would need to add some stocks to infra
    market_data[Nifty_INFRA]["sector_ids"].append(50)  # Reliance
    market_data[Nifty_INFRA]["sector_ids"].append(51)  # NTPC
    market_data[Nifty_INFRA]["sector_ids"].append(52)  # Powergrid

    # Save the skeleton of market data to pickle file
    save_dict_to_file(market_data, "market_data.pkl")
    save_dict_to_file(indicator_data, "indicator_data.pkl")


# load market data from the pickle file and save stock data to csv file by querying mongodb
# def loop_market_data_generate_df_save_csv(
#     db, market_data, start_date_15min, start_date_5min, start_date_3min, end_date=None
# ):
#     market_data = load_dict_from_file("market_data.pkl")

#     for stock_id, stock in market_data.items():
#         stock_symbol = stock["symbol"]
#         # We do not have intraday data for this stock hence need to comment
#         if stock_symbol == "VBL" or stock_symbol == "IDFC" or stock_symbol == "DRREDDY":
#             continue

#         df_1min = get_stock_data(db, stock_symbol, start_date_15min, end_date)
#         df_3_days = df_1min[df_1min.index.date >= start_date_5min]
#         df_2_days = df_1min[df_1min.index.date >= start_date_3min]

#         df_15min = resample_stock_data(df_1min)
#         df_5min = resample_stock_data(df_3_days, period="5min")
#         df_3min = resample_stock_data(df_2_days, period="3min")

#         # Need to combine this with intraday levels.
#         df_high_levels, df_low_levels = calculate_daily_levels(df_15min)

#         # Compute average of high, low and close for 15, 5, 3 min time frame
#         calculate_hlc3(df_15min, df_5min, df_3min)

#         # Calculate Supertrend for all periods and time frames
#         calculate_supertrend(df_15min, df_5min, df_3min)

#         calculate_ema(df_15min, df_5min, df_3min)

#         calculate_rsi(df_15min, df_5min, df_3min)
#         df_15min = df_15min.dropna()
#         df_5min = df_5min.dropna()
#         df_3min = df_3min.dropna()

#         calculate_ppo(df_15min, df_5min, df_3min)

#         calculate_roc(df_15min, df_5min, df_3min)

#         df_15min = df_15min.dropna()
#         df_5min = df_5min.dropna()
#         df_3min = df_3min.dropna()

#         df_15min_swing_levels = calculate_supertrend_levels(df_15min)
#         df_5min_swing_levels = calculate_supertrend_levels(df_5min)
#         df_3min_swing_levels = calculate_supertrend_levels(df_3min)

#         folder_path_15min = "ohlc_csv/15/ohlc/"
#         folder_path_15min_swing_levels = "ohlc_csv/15/swing_levels/"

#         folder_path_5min = "ohlc_csv/5/ohlc/"
#         folder_path_5min_swing_levels = "ohlc_csv/5/swing_levels/"

#         folder_path_3min = "ohlc_csv/3/ohlc/"
#         folder_path_3min_swing_levels = "ohlc_csv/3/swing_levels/"

#         folder_high_levels = "levels_csv/highs/"
#         folder_low_levels = "levels_csv/lows/"

#         save_to_csv(df_15min, f"{stock_id}.csv", folder_path_15min)
#         save_to_csv(
#             df_15min_swing_levels, f"{stock_id}.csv", folder_path_15min_swing_levels
#         )

#         save_to_csv(df_5min, f"{stock_id}.csv", folder_path_5min)
#         save_to_csv(
#             df_5min_swing_levels, f"{stock_id}.csv", folder_path_5min_swing_levels
#         )

#         save_to_csv(df_3min, f"{stock_id}.csv", folder_path_3min)
#         save_to_csv(
#             df_3min_swing_levels, f"{stock_id}.csv", folder_path_3min_swing_levels
#         )

#         save_to_csv(df_high_levels, f"{stock_id}.csv", folder_high_levels)
#         save_to_csv(df_low_levels, f"{stock_id}.csv", folder_low_levels)


# We have a fully populated market data object which is used to fill swing high and lows.
def loop_market_data_gen_swing_df_save_csv(market_data):
    for stock_id, stock in market_data.items():

        df_15min = stock[15]["ohlc"]
        df_5min = stock[5]["ohlc"]
        df_3min = stock[3]["ohlc"]

        df_15min_swing_levels = calculate_supertrend_levels(df_15min)
        df_5min_swing_levels = calculate_supertrend_levels(df_5min)
        df_3min_swing_levels = calculate_supertrend_levels(df_3min)

        folder_path_15min_swing_levels = "ohlc_csv/15/swing_levels/"
        folder_path_5min_swing_levels = "ohlc_csv/5/swing_levels/"
        folder_path_3min_swing_levels = "ohlc_csv/3/swing_levels/"

        save_to_csv(
            df_15min_swing_levels, f"{stock_id}.csv", folder_path_15min_swing_levels
        )

        save_to_csv(
            df_5min_swing_levels, f"{stock_id}.csv", folder_path_5min_swing_levels
        )

        save_to_csv(
            df_3min_swing_levels, f"{stock_id}.csv", folder_path_3min_swing_levels
        )


def load_csv_to_dataframe(folder_path, file_name, is_date_index=False):
    # Construct the full file path
    # file_path = os.path.join(folder_path, file_name)
    file_path = folder_path + file_name

    # Read the CSV file into a pandas DataFrame
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        return None

    if is_date_index:
        # Convert the 'datetime' column to datetime type
        df["datetime"] = pd.to_datetime(df["datetime"])

        # Set the 'datetime' column as the index
        df.set_index("datetime", inplace=True)

    return df


# load market data from the pickle file and load ohlc values for 15, 5, 3 min time frames from folders ohlc_csv
def retrieve_market_data(is_swing_level_loaded=True):
    market_data = load_dict_from_file("market_data.pkl")

    for stock_id, stock in market_data.items():

        # We do not have intraday data for this stock hence need to comment
        if stock["symbol"] == "VBL" or stock["symbol"] == "IDFC":
            continue

        # # We are first only testing if Nifty 50 levels are getting rendered correctly
        folder_path_15min = "ohlc_csv/15/ohlc/"
        folder_path_15min_swing_levels = "ohlc_csv/15/swing_levels/"

        folder_path_5min = "ohlc_csv/5/ohlc/"
        folder_path_5min_swing_levels = "ohlc_csv/5/swing_levels/"

        folder_path_3min = "ohlc_csv/3/ohlc/"
        folder_path_3min_swing_levels = "ohlc_csv/3/swing_levels/"

        folder_high_levels = "levels_csv/highs/"
        folder_low_levels = "levels_csv/lows/"

        file_name = f"{stock_id}.csv"

        stock["highs"] = load_csv_to_dataframe(folder_high_levels, file_name)
        stock["lows"] = load_csv_to_dataframe(folder_low_levels, file_name)
        stock[15]["ohlc"] = load_csv_to_dataframe(
            folder_path_15min, file_name, is_date_index=True
        )
        stock[5]["ohlc"] = load_csv_to_dataframe(
            folder_path_5min, file_name, is_date_index=True
        )
        stock[3]["ohlc"] = load_csv_to_dataframe(
            folder_path_3min, file_name, is_date_index=True
        )

        # Since we are initially only testing Nifty 50 and 15 min time frame we do not load other timeframes and swing high and low.
        if is_swing_level_loaded:
            stock[15]["swing_levels"] = load_csv_to_dataframe(
                folder_path_15min_swing_levels, file_name, is_date_index=True
            )

            stock[5]["swing_levels"] = load_csv_to_dataframe(
                folder_path_5min_swing_levels, file_name, is_date_index=True
            )

            stock[3]["swing_levels"] = load_csv_to_dataframe(
                folder_path_3min_swing_levels, file_name, is_date_index=True
            )

    return market_data


def main():

    # # The below block of code only when we would like to load the market_data with 1 min data for 15, 5, and 3 min time frame
    # # Initialize empty dictionary
    market_data = {}
    client = connect_to_mongodb()
    db = client["nse_stock_data"]

    # # Will need this only if we have added more stocks or changed the schema market_data.pkl
    df_stock_master = get_stock_master_data(db)
    # # If we change the schema of market data then we need to invoke this method.
    set_market_data(df_stock_master, market_data)
    return
    # end_date_str = "2024-04-02"
    # end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    # end_date = end_date.replace(tzinfo=pytz.UTC)
    # query = {"datetime": {"$lte": end_date}}
    # df_trading_days = get_trading_days_data(db, limit=15, query=query)
    df_trading_days = get_trading_days_data(db)

    # The data frame df_trading_days is ordered in descending order. Newest first
    # Since we have set a limit of 10 records the last record is from 10 trading days back.
    # start_date_15min = df_trading_days.iloc[0]["date"]

    # end_date = df_trading_days.iloc[-1]["date"]
    # start_date_5min = pd.Timestamp(df_trading_days.iloc[8]["date"]).date()
    # start_date_3min = pd.Timestamp(df_trading_days.iloc[10]["date"]).date()
    # print(start_date_15min, start_date_5min, start_date_3min, end_date)
    # return

    start_date_15min = df_trading_days.iloc[-1]["date"]
    start_date_5min = pd.Timestamp(df_trading_days.iloc[4]["date"]).date()
    start_date_3min = pd.Timestamp(df_trading_days.iloc[2]["date"]).date()

    # TODO: After saving the pickle file for combined_df and vector_indicators.
    # We get the next trading day from existing function and created combined_intraday_data by pivoting OHLC data for 1st April 2024.
    # We load the pickle file and we remove one day from swing levels for all time frames.
    # We load the highs and lows and add the previous days high and low and recompute.
    # We would save the combined_OHLC_df, combined_FULL_df, vector_indicators.pkl.
    # The reason for this is that for vectorizing indicator calls we need this.
    # Once we have set the chain for 1st April we run it in a loop always fetching the next day till we reach the end.
    # We can keep printing the completed days in terminal. If it is running too long we can cancel the process
    # and continue from where it left off.

    # We query the mongodb database and retreive ohlc for 15, 5, 3 min time frame
    # loop_market_data_generate_df_save_csv(
    #     db, market_data, start_date_15min, start_date_5min, start_date_3min, end_date
    # )

    loop_market_data_generate_df_save_csv(
        db, market_data, start_date_15min, start_date_5min, start_date_3min
    )

    client.close()

    # # Since we have change the file path to save market data and the schema.
    # market_data = retrieve_market_data(is_swing_level_loaded=False)
    # loop_market_data_gen_swing_df_save_csv(market_data)

    # Uncomment the following lines if you want to save the results to CSV
    # save_to_csv(df_nse_nifty_15min_supertrend, f"{stock_symbol}_supertrend.csv")
    # save_to_csv(df_nse_nifty_15min, f"{stock_symbol}.csv")


# Uncomment this function if you want to save results to CSV
def save_to_csv(df, filename, folder_path="csv/"):
    df.to_csv(folder_path + filename, index=True)


if __name__ == "__main__":
    main()
