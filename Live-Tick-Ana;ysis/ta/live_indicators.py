import asyncio
import math
import pickle
from typing import Dict, List, Callable, Optional, Tuple, Union
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
import os
import pickle
import logging
from enum import Enum

logging.basicConfig(
    filename="output.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_dict_from_file(filename):
    with open(filename, "rb") as file:
        loaded_dict = pickle.load(file)
    # print(f"Dictionary loaded from {filename}")
    # # Get the size of the loaded object in bytes
    # size_in_bytes = sys.getsizeof(market_data)
    # print(size_in_bytes)

    return loaded_dict


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

    print(f"Dictionary loaded from {file_path}")
    return pickle_object


class LevelType(Enum):
    """Enum for different stock market levels."""

    PREVIOUS_DAY_HIGH = 1
    PREVIOUS_DAY_CLOSE = 2
    PREVIOUS_DAY_LOW = 3
    DAY_OPEN = 4
    DAY_HIGH = 5
    DAY_LOW = 6
    UP_MOVE_START = 7
    UP_MOVE_PULLBACK = 8
    DOWN_MOVE_START = 9
    DOWN_MOVE_PULLBACK = 10
    SUPER_TREND_FIRST_PERIOD = 11
    SUPER_TREND_SECOND_PERIOD = 11


class EventType(Enum):
    """Enum for different stock market levels."""

    PREVIOUS_DAY_HIGH_BREAKOUT = 1
    PREVIOUS_DAY_HIGH_BREAKDOWN = 2
    PREVIOUS_DAY_CLOSE_BREAKOUT = 2
    PREVIOUS_DAY_CLOSE_BREAKDOWN = 2
    PREVIOUS_DAY_LOW_BREAKOUT = 3
    PREVIOUS_DAY_LOW_BREAKDOWN = 3
    DAY_OPEN_BREAKOUT = 4
    DAY_OPEN_BREAKDOWN = 4
    DAY_HIGH_BREAKOUT = 5
    DAY_HIGH_BREAKDOWN = 5
    DAY_LOW_BREAKOUT = 6
    DAY_LOW_BREAKDOWN = 6
    UP_MOVE_START_BREAKOUT = 7
    UP_MOVE_START_BREAKDOWN = 7
    DOWN_MOVE_START_BREAKOUT = 9
    DOWN_MOVE_START_BREAKDOWN = 9
    SUPER_TREND_FIRST_PERIOD_BREAKOUT = 11
    SUPER_TREND_SECOND_PERIOD_BREAKDOWN = 11


class Event:
    def __init__(self, event_type: str, data: Dict, stock_id: int, time_frame: int):
        self.event_type = event_type
        self.data = data
        self.stock_id = stock_id
        self.time_frame = time_frame


class BaseIndicator:
    def __init__(self, stock_id: int, time_frame: int):
        self.stock_id = stock_id
        self.time_frame = time_frame
        self.message = (
            ""  # This would be added when we raise a event in inherited class
        )
        self.raise_event = False
        self.listeners: List[Callable] = []
        # When we compare the previous and current value. For supertrend we need value of both band and supertrend
        self.event_data: Optional[
            Union[Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]]
        ] = None
        self.value = None

    def add_listener(self, callback: Callable):
        self.listeners.append(callback)

    async def notify(self, event: Event):
        for listener in self.listeners:
            await listener(event)

    def update_event_data(self, old_value):
        self.event_data = (old_value, self.value)

    def __getstate__(self):
        # This method is called when pickling
        state = self.__dict__.copy()
        # Don't pickle the listeners
        state["listeners"] = []
        return state

    # Base class method call is to reset raise event flag and message and take care of edge cases.
    # This is not called in the swing level class as this is not a technical indicator.
    def should_raise_event(
        self,
    ):
        # First reset the message and raise event flag and then execute the logic to raise event
        self.message = ""
        self.raise_event = False
        # Initialize the event data to default value the first time around.
        if self.event_data is None:
            self.event_data = (self.value, self.value)


class EMA(BaseIndicator):
    """Exponential Moving Average"""

    def __init__(
        self, stock_id: int, time_frame: int, period: int, smoothing_factor: int = 2
    ):
        # Added for raising event
        super().__init__(stock_id, time_frame)
        self.period = period
        self.smoothing_factor = smoothing_factor
        self.mult = smoothing_factor / (1 + period)
        self.points = deque(maxlen=period + 1)
        self.datetime = None

    def initialize(self, points, datetime):
        self.points = points
        self.datetime = datetime

    # Here we figure out if we have to raise a event
    def should_raise_event(
        self,
    ):
        # First reset the message and raise event flag and then execute the logic to raise event
        super().should_raise_event()

    async def update(self, candle: dict, datetime: datetime = None):
        point: float = candle["hlc3"]
        self.points.append(point)
        old_value = self.value

        if datetime is not None:
            self.datetime = datetime
        if len(self.points) == self.period:
            self.value = np.mean(self.points)  # Simple SMA
        elif len(self.points) > self.period:
            self.value = (point * self.mult) + (self.value * (1 - self.mult))

        # Set the old value and the changed value in base class
        super().update_event_data(old_value)
        # After the update based on a condition we may raise an event.
        self.should_raise_event()
        if self.raise_event:
            event = Event(
                "EMA9_UPDATE",  # This is the type of the event
                {  # This has the stock id, time frame and the message
                    "stock_id": self.stock_id,
                    "time_frame": self.time_frame,
                    "message": self.message,  # This property is defined in base class
                },
            )
            await self.notify(event)


class TRANGE:
    """True Range"""

    def __init__(self):
        self.prev_close = None
        self.value = None
        self.datetime = None

    def compute(self, candle):
        if self.prev_close is None:
            return candle["high"] - candle["low"]
        else:
            return max(
                candle["high"] - candle["low"],
                abs(candle["high"] - self.prev_close),
                abs(candle["low"] - self.prev_close),
            )

    def update(self, candle, datetime: datetime = None):
        if datetime is not None:
            self.datetime = datetime
        self.value = self.compute(candle)
        self.prev_close = candle["close"]
        return self.value


class ATR:
    """Average True Range"""

    def __init__(self, period, candles=None):
        self.period = period
        self.period_1 = period - 1
        self.TR = TRANGE()
        self.datetime = None
        if candles is None:
            self.atr = 0  # initialised to 0, because values are added to it
            self.value = None
            self.count = 0

    def compute(self, candle):
        tr = self.TR.compute(candle)
        if self.count < self.period:
            return None
        elif self.count == self.period:
            return (self.atr + tr) / self.period
        else:
            return (self.atr * self.period_1 + tr) / self.period

    def update(self, candle, datetime: datetime = None):
        if datetime is not None:
            self.datetime = datetime
        self.count += 1
        tr = self.TR.update(candle)
        if self.count < self.period:
            self.atr += tr
            return None
        if self.count == self.period:
            self.atr += tr
            self.atr /= self.period
        else:
            self.atr = (self.atr * self.period_1 + tr) / self.period
        self.value = self.atr
        return self.value


class SuperTrend(BaseIndicator):
    def __init__(self, stock_id: int, time_frame: int, atr_length, factor):
        super().__init__(stock_id, time_frame)
        self.factor = factor
        self.super_trend = 1

        self.ATR = ATR(atr_length)
        self.lower_band = None
        self.upper_band = None
        self.final_band = None
        self.value = (self.super_trend, self.final_band)  # direction, value
        self.datetime = None

    def initialize(
        self, lower_band, upper_band, final_band, datetime, super_trend, ATR
    ):
        self.lower_band = lower_band
        self.upper_band = upper_band
        self.final_band = final_band
        self.datetime = datetime
        self.ATR = ATR
        self.super_trend = super_trend
        self.value = (self.super_trend, self.final_band)

    def compute(self, candle):
        median = round((candle["high"] + candle["low"]) / 2, 4)
        atr = self.ATR.compute(candle)
        if atr is None:
            return None, None
        _fatr = self.factor * atr
        basic_upper_band = round(median + _fatr, 4)
        basic_lower_band = round(median - _fatr, 4)
        super_trend = self.super_trend
        if self.super_trend == 1:
            upper_band = basic_upper_band
            lower_band = (
                max(basic_lower_band, self.lower_band)
                if self.lower_band is not None
                else basic_lower_band
            )
            if candle["close"] <= self.lower_band:
                super_trend = -1
        else:
            lower_band = basic_lower_band
            upper_band = (
                min(basic_upper_band, self.upper_band)
                if self.upper_band is not None
                else basic_upper_band
            )
            if candle["close"] >= self.upper_band:
                super_trend = 1
        if super_trend == 1:
            final_band = lower_band
        else:
            final_band = upper_band
        return (super_trend, final_band)

    # Here we figure out if we have to raise a event
    def should_raise_event(
        self,
    ):
        # First reset the message and raise event flag and then execute the logic to raise event
        self.message = ""
        self.raise_event = False

    async def update(self, candle, datetime: datetime = None):
        old_value = self.value
        if datetime is not None:
            self.datetime = datetime
        median = round((candle["high"] + candle["low"]) / 2, 4)
        atr = self.ATR.update(candle)
        if atr is None:
            return None, None
        basic_upper_band = round(median + self.factor * atr, 4)
        basic_lower_band = round(median - self.factor * atr, 4)
        if self.super_trend == 1:
            self.upper_band = basic_upper_band
            self.lower_band = (
                max(basic_lower_band, self.lower_band)
                if self.lower_band is not None
                else basic_lower_band
            )
            if candle["close"] <= self.lower_band:
                self.super_trend = -1
        else:
            self.lower_band = basic_lower_band
            self.upper_band = (
                min(basic_upper_band, self.upper_band)
                if self.upper_band is not None
                else basic_upper_band
            )
            if candle["close"] >= self.upper_band:
                self.super_trend = 1

        if self.super_trend == 1:
            self.final_band = self.lower_band
        else:
            self.final_band = self.upper_band

        self.value = (self.super_trend, self.final_band)
        # Set the old value and the changed value in base class
        super().update_event_data(old_value)
        # After the update based on a condition we may raise an event.
        self.should_raise_event()
        if self.raise_event:
            event = Event(
                "SUPERTREND_UPDATE",  # This is the type of the event
                {  # This has the stock id, time frame and the message
                    "stock_id": self.stock_id,
                    "time_frame": self.time_frame,
                    "message": self.message,  # This property is defined in base class
                },
            )
            await self.notify(event)


class PPO:
    """Percentage Price Oscillator: We have to pass both the ema 9, 21 or ema9, hlc3 or rsi, smoothed MA"""

    def __init__(
        self,
    ):
        self.datetime = None

    @staticmethod
    def compute(fast_indicator: float, slow_indicator: float):
        if (
            fast_indicator is not None
            and slow_indicator is not None
            and slow_indicator != 0.0
        ):
            ppo = ((fast_indicator - slow_indicator) / slow_indicator) * 10000
            return ppo
        else:
            return 0.0


class SwingLevel(BaseIndicator):
    def __init__(
        self,
        stock_id: int,
        time_frame: int,
    ):
        super().__init__(stock_id, time_frame)
        self.supertrend_value = "st_fpvalue"
        self.supertrend_direction = "st_fpdirection"
        self.first_period = {
            "high_till_now": 0,
            "high_till_now_datetime": None,
            "low_till_now": 0,
            "low_till_now_datetime": None,
            "direction": 0,
            "value": 0,
            "flat_count": 0,
        }
        self.dict_start_up_move = {
            "datetime": 0,
            "support": 0,
            "resistance": 0,
            "is_up_move": 0,
            "is_up_trend": 0,
            "level": 0,
            "trigger_datetime": 0,
        }
        self.dict_start_down_move = {
            "datetime": 0,
            "support": 0,
            "resistance": 0,
            "is_down_move": 0,
            "is_up_move": 0,
            "is_up_trend": 0,
            "level": 0,
            "trigger_datetime": 0,
        }
        self.df_swing_levels = pd.DataFrame(
            columns=[
                "datetime",
                "is_up_trend",
                "is_up_move",
                "level",
                "trigger_datetime",
            ]
        )
        self.df_swing_levels.set_index("datetime", inplace=True)

    def initialize(
        self, first_period, dict_start_up_move, dict_start_down_move, df_swing_levels
    ):
        self.first_period = first_period
        self.dict_start_up_move = dict_start_up_move
        self.dict_start_down_move = dict_start_down_move
        self.df_swing_levels = df_swing_levels

    def set_period_values(self, candle, datetime_index):
        if (
            self.first_period["high_till_now"] == 0
            or self.first_period["high_till_now"] < candle["high"]
        ):
            self.first_period["high_till_now"] = candle["high"]
            self.first_period["high_till_now_datetime"] = datetime_index

        if (
            self.first_period["low_till_now"] == 0
            or self.first_period["low_till_now"] > candle["low"]
        ):
            self.first_period["low_till_now"] = candle["low"]
            self.first_period["low_till_now_datetime"] = datetime_index

        self.first_period["direction"] = candle[self.supertrend_direction]
        self.first_period["value"] = candle[self.supertrend_value]

    def reset_swing_levels(
        self,
        df_swing_levels=None,
        first_period=None,
        dict_start_up_move=None,
        dict_start_down_move=None,
    ):
        # For 28th March 2024 reset all values
        if df_swing_levels is None:
            self.df_swing_levels = self.df_swing_levels.iloc[0:0]
            self.first_period = {
                "high_till_now": 0,
                "high_till_now_datetime": None,
                "low_till_now": 0,
                "low_till_now_datetime": None,
                "direction": 0,
                "value": 0,
                "flat_count": 0,
            }
            self.dict_start_up_move = {
                "datetime": 0,
                "support": 0,
                "resistance": 0,
                "is_up_move": 0,
                "is_up_trend": 0,
                "level": 0,
                "trigger_datetime": 0,
            }
            self.dict_start_down_move = {
                "datetime": 0,
                "support": 0,
                "resistance": 0,
                "is_down_move": 0,
                "is_up_move": 0,
                "is_up_trend": 0,
                "level": 0,
                "trigger_datetime": 0,
            }
        # For dates greater than 28th March 2024 reset it to the previous day.
        else:
            self.df_swing_levels = df_swing_levels.copy()
            self.first_period = first_period
            self.dict_start_up_move = dict_start_up_move
            self.dict_start_down_move = dict_start_down_move

    def create_new_start_of_upmove(self, trigger_datetime, is_up_move=True):
        new_start_of_upmove_row = {
            "datetime": 0,
            "support": 0,
            "resistance": 0,
            "is_up_move": 0,
            "is_up_trend": 0,
            "level": 0,
            "trigger_datetime": 0,
        }

        if is_up_move:
            new_start_of_upmove_row["datetime"] = self.first_period[
                "low_till_now_datetime"
            ]
            new_start_of_upmove_row["level"] = self.first_period["low_till_now"]
        else:
            new_start_of_upmove_row["datetime"] = self.first_period[
                "high_till_now_datetime"
            ]
            new_start_of_upmove_row["level"] = self.first_period["high_till_now"]

        new_start_of_upmove_row["support"] = self.first_period["low_till_now"]
        new_start_of_upmove_row["resistance"] = self.first_period["high_till_now"]
        new_start_of_upmove_row["is_up_move"] = is_up_move
        new_start_of_upmove_row["is_up_trend"] = True
        new_start_of_upmove_row["trigger_datetime"] = trigger_datetime

        return new_start_of_upmove_row

    def create_new_start_of_downmove(
        self,
        trigger_datetime,
        is_down_move=True,
    ):
        new_start_of_downmove_row = {
            "datetime": 0,
            "support": 0,
            "resistance": 0,
            "is_down_move": 0,
            "is_up_move": 0,
            "is_up_trend": 0,
            "level": 0,
            "trigger_datetime": 0,
        }

        if is_down_move:
            new_start_of_downmove_row["datetime"] = self.first_period[
                "high_till_now_datetime"
            ]
            new_start_of_downmove_row["level"] = self.first_period["high_till_now"]
            new_start_of_downmove_row["is_up_move"] = False
        else:
            new_start_of_downmove_row["datetime"] = self.first_period[
                "low_till_now_datetime"
            ]
            new_start_of_downmove_row["level"] = self.first_period["low_till_now"]
            new_start_of_downmove_row["is_up_move"] = True

        new_start_of_downmove_row["support"] = self.first_period["low_till_now"]
        new_start_of_downmove_row["resistance"] = self.first_period["high_till_now"]
        new_start_of_downmove_row["is_down_move"] = is_down_move
        new_start_of_downmove_row["is_up_trend"] = False
        new_start_of_downmove_row["trigger_datetime"] = trigger_datetime

        return new_start_of_downmove_row

    def initialize_period_dict(self, bar_datetime, candle):
        self.first_period = {
            "high_till_now": candle["high"],
            "high_till_now_datetime": bar_datetime,
            "low_till_now": candle["low"],
            "low_till_now_datetime": bar_datetime,
            "direction": candle[self.supertrend_direction],
            "value": candle[self.supertrend_value],
            "flat_count": 0,
        }

    def should_insert_trend_pullback_row(self, is_start_of_up_move=False):
        should_insert = False

        if is_start_of_up_move:
            # Since we cover a long duration we ignore swing points till we have a crossover in supertrend.
            if self.dict_start_up_move["level"] == 0:
                return should_insert

            if self.dict_start_up_move["is_up_move"]:
                should_insert = True
        else:
            if self.dict_start_up_move["level"] == 0:
                return should_insert

            if self.dict_start_down_move["is_down_move"]:
                should_insert = True

        return should_insert

    # This will check if close is higher than resistance of last record and there is a pullback
    def should_insert_trend_continuation_row(
        self, bar_high_or_low, is_start_of_up_move=False
    ):
        should_insert = False

        if is_start_of_up_move:
            # Since we cover a long duration we ignore swing points till we have a crossover in supertrend.
            if self.dict_start_up_move["level"] == 0:
                return should_insert

            if (
                not self.dict_start_up_move["is_up_move"]
                and bar_high_or_low > self.dict_start_up_move["resistance"]
            ):
                should_insert = True
        else:
            if self.dict_start_up_move["level"] == 0:
                return should_insert

            if (
                not self.dict_start_down_move["is_down_move"]
                and bar_high_or_low < self.dict_start_down_move["support"]
            ):
                should_insert = True

        return should_insert

    # Here we figure out if we have to raise a event
    def should_raise_event(
        self,
    ):
        # First reset the message and raise event flag and then execute the logic to raise event
        super().should_raise_event()
        # We can query the df_swing_level and datetime variable to raise event.

    def remove_excessive_swing_levels(
        self,
    ):
        max_date = self.df_swing_levels["trigger_datetime"].max()
        # We are setting the time frame to number of days.
        cutoff_date = max_date - pd.Timedelta(days=self.time_frame)

        self.df_swing_levels = self.df_swing_levels[
            self.df_swing_levels["trigger_datetime"] > cutoff_date
        ]
        # Even now we have excessive swing levels then just crop.
        if len(self.df_swing_levels) > 100:
            self.df_swing_levels = self.df_swing_levels.sort_values(
                "trigger_datetime", ascending=True
            ).head(100)

    async def update(self, datetime_index, candle):
        add_up_move = False
        add_down_move = False
        if self.first_period["direction"] == 0:
            self.initialize_period_dict(datetime_index, candle)

        # There is two possibilities first is crossover to down trend and second is crossover to up trend
        elif candle[self.supertrend_direction] != self.first_period["direction"]:
            # 1. We have to save to either start of up move or start of down move
            if candle[self.supertrend_direction] == 1:
                # We have crossover to an uptrend
                new_start_of_upmove_row = self.create_new_start_of_upmove(
                    datetime_index,
                    is_up_move=True,
                )
                add_up_move = True
                self.dict_start_up_move = new_start_of_upmove_row
            else:
                # We have crossover to an downtrend
                new_start_of_downmove_row = self.create_new_start_of_downmove(
                    datetime_index,
                    is_down_move=True,
                )
                add_down_move = True
                self.dict_start_down_move = new_start_of_downmove_row

            # reset the period to the current candle
            self.initialize_period_dict(datetime_index, candle)
        else:
            # This condition is to add pullback when first period super trend flattens
            if candle[self.supertrend_value] == self.first_period["value"]:
                if self.first_period["flat_count"] == 0:
                    self.set_period_values(candle, datetime_index)
                    self.first_period["flat_count"] = 1
                elif self.first_period["flat_count"] == 1:
                    # 1. If this is start of down move in a uptrend then we have to check if there is a prior record of start of down move.
                    # 2. If no prior start of down move We have to save to either start of up move or start of down move
                    if candle[self.supertrend_direction] == 1:
                        # This is a pullback in a up move
                        # If the last record has column is_up_move as True then only we insert a record
                        should_insert = self.should_insert_trend_pullback_row(
                            is_start_of_up_move=True
                        )

                        if should_insert:
                            new_start_of_upmove_row = self.create_new_start_of_upmove(
                                datetime_index,
                                is_up_move=False,
                            )
                            new_start_of_upmove_row["support"] = candle["low"]
                            add_up_move = True
                            self.dict_start_up_move = new_start_of_upmove_row
                    else:
                        # This is a pullback in a down move
                        should_insert = self.should_insert_trend_pullback_row(
                            is_start_of_up_move=False
                        )

                        if should_insert:
                            new_start_of_downmove_row = (
                                self.create_new_start_of_downmove(
                                    datetime_index,
                                    is_down_move=False,
                                )
                            )
                            new_start_of_downmove_row["resistance"] = candle["high"]
                            add_down_move = True
                            self.dict_start_down_move = new_start_of_downmove_row

                    self.initialize_period_dict(datetime_index, candle)
                    self.first_period["flat_count"] = 2
                else:
                    # We have already registered the pullback and now we either have crossover or continuation of trend
                    self.set_period_values(candle, datetime_index)
                    self.first_period["flat_count"] += 1
            else:
                # This part of code gets executed when price is in trend continuation.
                if candle[self.supertrend_direction] == 1:
                    # Since we cover a long duration we ignore swing points till we have a crossover.
                    # Verifying prior record is a pullback
                    should_insert_record = self.should_insert_trend_continuation_row(
                        candle["high"],
                        is_start_of_up_move=True,
                    )
                    if should_insert_record:
                        new_start_of_upmove_row = self.create_new_start_of_upmove(
                            datetime_index,
                            is_up_move=True,
                        )
                        add_up_move = True
                        self.dict_start_up_move = new_start_of_upmove_row

                        self.initialize_period_dict(datetime_index, candle)
                    else:
                        self.set_period_values(candle, datetime_index)
                else:
                    should_insert_record = self.should_insert_trend_continuation_row(
                        candle["low"],
                        is_start_of_up_move=False,
                    )
                    if should_insert_record:
                        new_start_of_downmove_row = self.create_new_start_of_downmove(
                            datetime_index,
                            is_down_move=True,
                        )
                        add_down_move = True
                        self.initialize_period_dict(datetime_index, candle)
                        self.dict_start_down_move = new_start_of_downmove_row
                    else:
                        self.set_period_values(candle, datetime_index)
                self.first_period["flat_count"] = 0

        # new_start_of_upmove_row, new_start_of_downmove_row
        if add_up_move == False and add_down_move == False:
            return

        if add_up_move == True:
            df = pd.DataFrame([new_start_of_upmove_row])
        elif add_down_move == True:
            df = pd.DataFrame([new_start_of_downmove_row])
        else:
            return

        df.set_index("datetime", inplace=True)
        columns_to_add = ["is_up_trend", "is_up_move", "level", "trigger_datetime"]
        df = df[columns_to_add]
        self.df_swing_levels = pd.concat(
            [self.df_swing_levels if not self.df_swing_levels.empty else None, df],
        )
        self.df_swing_levels.sort_index(inplace=True)
        self.should_raise_event()
        if self.raise_event:
            event = Event(
                "SWING_LEVELS_UPDATE",  # This is the type of the event
                {  # This has the stock id, time frame and the message
                    "stock_id": self.stock_id,
                    "time_frame": self.time_frame,
                    "message": self.message,  # This property is defined in base class
                },
            )
            await self.notify(event)
        # Limit the number of swing levels.
        if len(self.df_swing_levels) > 100:
            self.remove_excessive_swing_levels()


class PearsonsCorrelation(BaseIndicator):
    def __init__(
        self,
        stock_id: int,
        time_frame: int,
        window_size,
    ):
        # Added for raising event
        super().__init__(stock_id, time_frame)

        self.window_size = window_size
        self.y_values = deque(maxlen=window_size)

    # Here we figure out if we have to raise a event
    def should_raise_event(
        self,
    ):
        # First reset the message and raise event flag and then execute the logic to raise event
        super().should_raise_event()

    def update(self, y):
        self.y_values.append(y)

    def pearson_correlation(self):
        n = len(self.y_values)
        if n < 2:
            return 0.0

        # Generate x as indices
        x = np.arange(n)

        # Compute means
        x_mean = np.mean(x)
        y_mean = np.mean(self.y_values)

        # Compute covariance and standard deviations
        cov_xy = np.sum((x - x_mean) * (np.array(self.y_values) - y_mean))
        x_std = np.sqrt(np.sum((x - x_mean) ** 2))
        y_std = np.sqrt(np.sum((np.array(self.y_values) - y_mean) ** 2))

        # Avoid division by zero
        if x_std == 0 or y_std == 0:
            return 0.0

        return round(cov_xy / (x_std * y_std), 2)

    def r_squared(self):
        correlation = self.pearson_correlation()
        return round(correlation**2, 2)


class StandardDeviation(BaseIndicator):
    def __init__(
        self,
        stock_id: int,
        time_frame: int,
        window_size,
    ):
        # Added for raising event
        super().__init__(stock_id, time_frame)

        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.sum = 0.0
        self.sum_squared = 0.0

    # Here we figure out if we have to raise a event
    def should_raise_event(
        self,
    ):
        # First reset the message and raise event flag and then execute the logic to raise event
        super().should_raise_event()

    def update(self, value):
        # If window is full, remove the oldest value
        if len(self.values) == self.window_size:
            old_value = self.values[0]
            self.sum -= old_value
            self.sum_squared -= old_value**2

        # Add new value
        self.values.append(value)
        self.sum += value
        self.sum_squared += value**2

    def mean(self):
        # Use the adjust_and_mean function to calculate the mean
        # Find the minimum value in the deque
        min_val = min(self.values)

        # If the minimum value is negative, adjust the data
        if min_val < 0:
            adjusted_data = [x - min_val for x in self.values]
        else:
            adjusted_data = list(self.values)

        # Calculate the mean of the adjusted data
        mean_val = np.mean(adjusted_data)

        return mean_val

    def standard_deviation(self):
        # Use sample standard deviation (n-1 in denominator)
        if len(self.values) < 2:
            return 0.0

        n = len(self.values)
        variance = (self.sum_squared - (self.sum**2 / n)) / (n - 1)
        return math.sqrt(
            max(0, variance)
        )  # Protect against tiny negative values due to float precision

    def coefficient_of_variation(self):
        # Calculate the coefficient of variation
        mean_val = self.mean()
        std_dev = self.standard_deviation()
        if mean_val == 0:
            return 0.0

        return round((std_dev / mean_val) * 100, 2)


# This class has all the levels raised by Swing Level, Super trend, High, Low and Close
class Levels(BaseIndicator):
    # TODO: Whenever a swing level is created it will update this class for the stock id and time frame.
    # We need to update these levels. This would also happen when we have constant value super trend
    # During init we would have to add the previous days levels for this stock.
    def __init__(
        self,
        stock_id: int,
        time_frame: int,
    ):
        # Added for raising event
        super().__init__(stock_id, time_frame)
        # Whether a level is support or resistance would depend on the current hlc3 value.
        # If the level is DH or DL it could change frequently
        self.df_levels = pd.DataFrame(
            columns=[
                "days_past",  # This is a integer column
                "time",  # This is %H:%M that is 12:30 column
                "level",  # This is the actual value and will not change
                "ppo_hlc3_level",  # This would be changed with every update of hlc3 value
                # level_int = LevelType.PREVIOUS_DAY_HIGH.value
                "level_type",  # This would be int converted from a enum class
            ]
        )

    # TODO: We have to build the event framework for this whenever there is a crossover we would have to raise an event.


# This event is raised mostly by level breakout and could be part of a pattern
class Events:
    # TODO: Have to implement the update class which would add data to this
    def __init__(
        self,
    ):
        # Whether a level is support or resistance would depend on the current hlc3 value.
        # If the level is DH or DL it could change frequently
        self.df_events = pd.DataFrame(
            columns=[
                "time",  # This is %H:%M that is 12:30 column
                # event_int = EventType.PREVIOUS_DAY_HIGH_BREAKOUT.value
                "event_type",  # This would be int converted from a enum class
                "event_data",  # While breakout of a level could be a reason. But it could be part of a pattern.
            ]
        )


class Live_Indicators:

    def __init__(
        self,
        time_frame: int,
        indicator_data: Dict[int, Dict[int, Dict[int, BaseIndicator]]],
    ):
        # This is fully loaded with previous days data for all stocks.
        self.indicator_data: Dict[int, Dict[int, Dict[int, BaseIndicator]]] = (
            indicator_data
        )

        # The column values of the dataframe. This is used during update of technical indicators.
        self.supertrend_first_period_value = "st_fpvalue"
        self.supertrend_first_period_direction = "st_fpdirection"
        self.supertrend_second_period_value = "st_spvalue"
        self.supertrend_second_period_direction = "st_spdirection"
        self.EMA9_column_name = "ema_9"
        self.ema50_column_name = "ema_50"
        self.ppo_hlc3_ema9_column_name = "ppo_hlc3ema9"
        self.ppo_hlc3_ema50_column_name = "ppo_hlc3ema50"
        # New columns
        self.pearsons_correlation_column_name = "p-corr"
        self.r_squared = "r-squared"
        self.mean = "mean"
        self.standard_deviation_column_name = "std_dev"
        self.coefficient_of_variation = "coeff_of_var"

        # The indicator id so that the dictionary remains lightweight.
        self.ema9 = 1
        self.ema50 = 2
        self.st_fp = 3
        self.st_sp = 4
        self.pearsons_correlation = 5
        self.standard_deviation = 6
        self.swing_levels = 7
        self.levels = 8
        self.events = 9

        self.datetime = None

        self.time_frame = time_frame
        self.supertrend_period_length = 10
        self.supertrend_first_period_multiplier = 2
        self.supertrend_second_period_multiplier = 3

        # This is likely to change
        self.df = self.create_empty_df()

    def create_empty_df(
        self,
    ):
        return pd.DataFrame(
            columns=[
                "stock_id",
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
                "ema_50",
                "ppo_hlc3ema9",
                "ppo_hlc3ema50",
                "p-corr",
                "r-squared",
                "mean",
                "std_dev",
                "coeff_of_var",
            ],
            dtype="float64",
        )

    async def handle_event(self, event: Event):
        print(f"Received event: {event.event_type}")
        # The event data is a JSON with stock_id, time_frame and message. Need to process it
        # By mapping the stock symbol with stock_id and string of time frame.
        # When we build FastAPI this would start stream push data and stop streaming.
        # TODO: Once we get an event it has to be seen in context of all indicators and if it is worthy.
        # Then we need to load it in the Event indicator dataframe.
        print(
            f"Event data: {event.data}, Event stock id: {event.stock_id}, Event time frame: {event.time_frame}"
        )

    async def update_row(self, stock_id, row):
        stock_id = int(stock_id)
        # For these stocks we do not have consistent data
        if stock_id == 23 or stock_id == 64 or stock_id == 82:
            return

        hlc3 = row["hlc3"]

        # We cannot do concurrent operations as values from one operation is used in others.
        ema9: EMA = self.indicator_data[stock_id][self.time_frame][self.ema9]
        await ema9.update(row, self.datetime)
        self.df.at[stock_id, self.EMA9_column_name] = round(ema9.value or 0, 2)

        ema21: EMA = self.indicator_data[stock_id][self.time_frame][self.ema21]
        await ema21.update(row, self.datetime)
        self.df.at[stock_id, self.EMA21_column_name] = round(ema21.value or 0, 2)

        ema50: EMA = self.indicator_data[stock_id][self.time_frame][self.ema50]
        await ema50.update(row, self.datetime)
        self.df.at[stock_id, self.ema50_column_name] = round(ema50.value or 0, 2)

        # Only the first period super trend value is required to get the swing levels.
        st_fp: SuperTrend = self.indicator_data[stock_id][self.time_frame][self.st_fp]
        await st_fp.update(row, self.datetime)
        self.df.at[stock_id, self.supertrend_first_period_value] = round(
            st_fp.final_band or 0, 2
        )
        self.df.at[stock_id, self.supertrend_first_period_direction] = st_fp.super_trend

        st_sp: SuperTrend = self.indicator_data[stock_id][self.time_frame][self.st_sp]
        await st_sp.update(row, self.datetime)
        self.df.at[stock_id, self.supertrend_second_period_value] = round(
            st_sp.final_band or 0, 2
        )
        self.df.at[stock_id, self.supertrend_second_period_direction] = (
            st_sp.super_trend
        )

        if ema9.value is not None and ema9.value != 0.0:
            self.df.at[stock_id, self.ppo_hlc3_ema9_column_name] = round(
                PPO.compute(hlc3, ema9.value) or 0, 2
            )

        if st_fp.final_band is not None and st_fp.final_band != 0.0:
            self.df.at[stock_id, self.ppo_hlc3st_fp_column_name] = round(
                PPO.compute(hlc3, st_fp.final_band) or 0, 2
            )

        if st_sp.final_band is not None and st_sp.final_band != 0.0:
            self.df.at[stock_id, self.ppo_hlc3st_sp_column_name] = round(
                PPO.compute(hlc3, st_sp.final_band) or 0, 2
            )

        # Since we update LinReg in 1 min time intervals we just pick the value and store it in dataframe.
        # Provided it is not null
        pearsons_correlation: PearsonsCorrelation = self.indicator_data[stock_id][
            self.time_frame
        ][self.pearsons_correlation]

        pearson_value = pearsons_correlation.pearson_correlation()
        if pearson_value is not None:
            self.df.at[stock_id, self.pearsons_correlation_column_name] = round(
                pearson_value or 0, 2
            )

        r_squared = pearsons_correlation.r_squared()
        if r_squared is not None:
            self.df.at[stock_id, self.r_squared] = round(r_squared or 0, 2)

        standard_deviation = self.indicator_data[stock_id][self.time_frame][
            self.standard_deviation
        ]

        mean = standard_deviation.mean()
        if mean is not None:
            self.df.at[stock_id, self.mean] = round(mean or 0, 2)

        std_dev = standard_deviation.standard_deviation()
        if std_dev is not None:
            self.df.at[stock_id, self.standard_deviation_column_name] = round(
                std_dev or 0, 2
            )

        cov = standard_deviation.coefficient_of_variation()
        if cov is not None:
            self.df.at[stock_id, self.coefficient_of_variation] = round(cov or 0, 2)

        # This is different in the sense that it does not populate df but maintains df in the class
        # We would need to raise the event and bubble it up.
        swing_levels: SwingLevel = self.indicator_data[stock_id][self.time_frame][
            self.swing_levels
        ]
        # We need the supertrend value and direction
        candle = self.df.loc[stock_id].to_dict()
        # Data row addition happens occasionally. But every function call updates internal state.
        await swing_levels.update(self.datetime, candle)
        # Swing levels are not added to df as we do not have swing levels at every bar.

    # Reset swing levels to empty for 28th March 2024 and to previous day for greater than 28th March 2024.
    def reset_swing_levels(
        self,
        indicator_data: Dict[int, Dict[int, Dict[int, BaseIndicator]]] = None,  # type: ignore
    ):
        # We have to ignore these stock ids
        # empty_stock_ids = set([23, 64, 82])
        # We have to update
        for stock_id, stock in self.indicator_data.items():
            # if stock_id in empty_stock_ids:
            #     continue
            # Get the swing levels for each stock id, time frame and indicator id
            swing_levels: SwingLevel = self.indicator_data[stock_id][self.time_frame][
                self.swing_levels  # -> This is indicator id for swing levels
            ]

            df_swing_levels = None
            first_period = None
            dict_start_up_move = None
            dict_start_down_move = None

            # We reset the values to the close of previous day.
            if indicator_data is not None:
                swing_levels_previous: SwingLevel = indicator_data[stock_id][
                    self.time_frame
                ][
                    self.swing_levels  # -> This is indicator id for swing levels
                ]
                df_swing_levels = swing_levels_previous.df_swing_levels
                first_period = swing_levels_previous.first_period
                dict_start_up_move = swing_levels_previous.dict_start_up_move
                dict_start_down_move = swing_levels_previous.dict_start_down_move

            swing_levels.reset_swing_levels(
                df_swing_levels, first_period, dict_start_up_move, dict_start_down_move
            )

    # We start from the first and update the self.datetime
    async def update_swing_level(self, full_df, intraday_datetime: datetime = None):
        # We already have all the technical indicators.
        self.df = full_df.copy()

        for stock_id, row in self.df.iterrows():
            # We have to check the time if the time is 9:15 then remove all swing levels greater than current time.
            swing_levels: SwingLevel = self.indicator_data[stock_id][self.time_frame][
                self.swing_levels
            ]
            # We need the supertrend value and direction
            candle = self.df.loc[stock_id].to_dict()
            # print(candle)
            # Data row addition happens occasionally. But every function call updates internal state.
            await swing_levels.update(intraday_datetime, candle)
            # return

    # This is called every 1 minute to update Linear Regression and Standard Deviation
    def update_stats_values(self, stock_id, new_value_y):
        stock_id = int(stock_id)
        # For these stocks we do not have consistent data
        if stock_id == 23 or stock_id == 64 or stock_id == 82:
            return

        # When the Vector Indicators is initialized then self.df is empty and it would throw an error.
        # Hence we do this check.
        if stock_id in self.df.index:
            ema_50_y = self.df.at[stock_id, self.ema50_column_name]
            ema_9_y = self.df.at[stock_id, self.EMA9_column_name]
        else:
            return

        if ema_50_y is None or ema_50_y == 0.0:
            return

        # Normalize the values so that we can compare Linear Regression Independent values
        value_y = PPO.compute(new_value_y, ema_50_y)

        pearsons_correlation: PearsonsCorrelation = self.indicator_data[stock_id][
            self.time_frame
        ][self.pearsons_correlation]
        # We are only computing independent values for pearsons correlation and r-squared
        pearsons_correlation.update(value_y)

        # Normalize the values so that we can compare Linear Regression Independent values
        value_y = PPO.compute(new_value_y, ema_9_y)

        standard_deviation: StandardDeviation = self.indicator_data[stock_id][
            self.time_frame
        ][self.standard_deviation]
        # Calculation of mean, standard deviation and coefficient of variation
        standard_deviation.update(value_y)

    # This is always called with 1 min values
    def update_stats(self, df_y, df_stock_master):
        # If we have missing data we need to stop trying to update the technical indicators
        open = df_y.get("open")
        high = df_y.get("high")
        low = df_y.get("low")
        close = df_y.get("close")

        if open is None or high is None or low is None or close is None:
            print("Missing column(s)")
            print("DataFrame Head:")
            print(df_y.head())
            print("\nDataFrame Tail:")
            print(df_y.tail())
            return
        else:
            # Assign the hlc3 column value. It is actually hl2c4 value
            # Since we are dealing with 1 min data we do not have hlc3 column.
            df_y["hlc3"] = round((high + low + (close * 2)) / 4, 2)

        # Reset the index of full_df to create a new DataFrame with the same structure as empty_df
        df_y = df_y.reset_index()

        # If we do not do this it will create a subtle error where the numbering would become stock id when we have for loop
        df_y.set_index("stock_id", inplace=True)
        df_y = df_y[["hlc3"]]

        # Here for each stock we compute the indicator value
        for stock_id, row in df_y.iterrows():
            stock_hlc3 = row["hlc3"]
            # logging.info(
            #     f"The sector id: {sector_id} for the stock id: {stock_id}, sector hlc3 value is {sector_hlc3}, stock hlc3 value is {stock_hlc3}"
            # )
            self.update_stats_values(
                stock_id,
                stock_hlc3,
            )

    # This would be called every min and this would update the Linear Regression and Standard Deviation computation
    async def update_1min(self, full_df):
        # If we have missing data we need to stop trying to update the technical indicators
        open = full_df.get("open")
        high = full_df.get("high")
        low = full_df.get("low")
        close = full_df.get("close")

        if open is None or high is None or low is None or close is None:
            print("Missing column(s)")
            print("DataFrame Head:")
            print(full_df.head())
            print("\nDataFrame Tail:")
            print(full_df.tail())
            return
        else:
            # Assign the hlc3 column value. It is actually hl2c4 value
            full_df["hlc3"] = round((high + low + (close * 2)) / 4, 2)

        # Reset the index of full_df to create a new DataFrame with the same structure as empty_df
        full_df = full_df.reset_index()

        # TODO: Add the code to update the Linear Regression and Standard Deviation.

    # df is a dataframe that has OHLC values need to add columns and then assign to instance df.
    async def update(self, full_df, trading_date: datetime = None):
        self.datetime = trading_date
        # If we have missing data we need to stop trying to update the technical indicators
        open = full_df.get("open")
        high = full_df.get("high")
        low = full_df.get("low")
        close = full_df.get("close")

        if open is None or high is None or low is None or close is None:
            print("Missing column(s)")
            print("DataFrame Head:")
            print(full_df.head())
            print("\nDataFrame Tail:")
            print(full_df.tail())
            return
        else:
            # Assign the hlc3 column value. It is actually hl2c4 value
            full_df["hlc3"] = round((high + low + (close * 2)) / 4, 2)

        # Reset the index of full_df to create a new DataFrame with the same structure as empty_df
        full_df = full_df.reset_index()

        # Fill the empty columns in full_df with 0.0
        for col in self.df.columns:
            if col not in full_df.columns:
                full_df[col] = 0.0

        # If we do not do this it will create a subtle error where the numbering would become stock id when we have for loop
        full_df.set_index("stock_id", inplace=True)
        # The instance df is now a empty copy for that time interval
        self.df = full_df.copy()

        # Here for each stock we compute the indicator value
        indicator_tasks = [
            asyncio.create_task(self.update_row(stock_id, row))
            for stock_id, row in self.df.iterrows()
        ]
        # We can concurrently execute indicator computation for all stocks.
        await asyncio.gather(*indicator_tasks)

    def __getstate__(self):
        # This method is called when pickling
        self.df = self.create_empty_df()
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        # This method is called when unpickling
        self.__dict__.update(state)
        # Reconnect event listeners
        for stock in self.indicator_data.values():
            for time_frame in stock.values():
                for indicator in time_frame.values():
                    indicator.add_listener(self.handle_event)
