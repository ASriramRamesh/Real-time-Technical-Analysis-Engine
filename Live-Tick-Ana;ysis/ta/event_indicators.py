import asyncio
import pickle
from typing import Dict, List, Callable, Optional, Tuple, Union
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
import os
import pickle
import logging

logging.basicConfig(
    filename="output.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# TODO: We have to figure out how to implement 1 min Linear Regression in all time frames.
# Also set the rolling window level for 3 min as 9, 5 min as 15 and 15 min as 45.
# The best approach is to keep a 1 min time but use it to update Linear Regression for 3, 5, 15 min.
# In the 3 min interval do not update but query the Linear Regression values.


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


class Event:
    def __init__(self, event_type: str, data: Dict):
        self.event_type = event_type
        self.data = data


class BaseIndicator:
    def __init__(self, stock_id: int, time_frame: int):
        self.stock_id = stock_id
        self.time_frame = time_frame
        self.message = (
            ""  # This would be added when we raise a event in inherited class
        )
        self.raise_event = False
        self.listeners: List[Callable] = []
        # When we compare the previous and current value
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

    # Base class method call is to reset raise event flad and message and take care of edge cases.
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


class RSI(BaseIndicator):
    """Relative Strength Index"""

    def __init__(self, stock_id: int, time_frame: int, period: int):
        super().__init__(stock_id, time_frame)
        self.period = period
        self._period_minus_1 = period - 1
        self._period_plus_1 = period + 1
        self.points = deque(maxlen=self._period_plus_1)
        self.losses = deque(maxlen=self._period_plus_1)
        self.gains = deque(maxlen=self._period_plus_1)
        self.avg_gain = None
        self.avg_loss = None
        self.rsi = None
        self.datetime = None

    # Here we figure out if we have to raise a event
    def should_raise_event(
        self,
    ):
        # First reset the message and raise event flag and then execute the logic to raise event
        self.message = ""
        self.raise_event = False

    async def update(self, candle: dict, datetime: datetime = None):
        point: float = candle["close"]
        self.points.append(point)
        old_value = self.value
        if datetime is not None:
            self.datetime = datetime
        if len(self.points) > 1:
            diff = self.points[-1] - self.points[-2]
            if diff >= 0:
                self.gains.append(diff)
                self.losses.append(0)
            else:
                self.gains.append(0)
                self.losses.append(-diff)

            if len(self.points) == self._period_plus_1:
                if self.avg_gain is None:
                    self.avg_gain = np.mean(self.gains)
                    self.avg_loss = np.mean(self.losses)
                else:
                    self.avg_gain = (
                        (self.avg_gain * (self._period_minus_1)) + self.gains[-1]
                    ) / self.period
                    self.avg_loss = (
                        (self.avg_loss * (self._period_minus_1)) + self.losses[-1]
                    ) / self.period
                rs = np.divide(
                    self.avg_gain,
                    self.avg_loss,
                    out=np.zeros_like(self.avg_gain),
                    where=self.avg_loss != 0,
                )
                # rs = self.avg_gain / self.avg_loss
                self.rsi = 100 - (100 / (1 + rs))
                self.value = self.rsi

        # Set the old value and the changed value in base class
        super().update_event_data(old_value)
        # After the update based on a condition we may raise an event.
        self.should_raise_event()
        if self.raise_event:
            event = Event(
                "RSI_UPDATE",  # This is the type of the event
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


class ROC(BaseIndicator):
    def __init__(
        self, indicator_str: str, stock_id: int, time_frame: int, period: int = 1
    ):
        super().__init__(stock_id, time_frame)
        self.period = period
        self.points = deque(maxlen=period + 1)
        self.datetime = None
        # This is for configuring the message while raising event
        self.study_type = indicator_str
        self.type = f"ROC_{indicator_str}_UPDATE"

    def compute(self, point: float):
        points = (list(self.points) + [float(point)])[-self.period :]
        if len(points) > self.period:
            return ((point - points[0]) / points[0]) * 10000
        return None

    # Here we figure out if we have to raise a event
    def should_raise_event(
        self,
    ):
        # First reset the message and raise event flag and then execute the logic to raise event
        self.message = ""
        self.raise_event = False

    async def update(self, point: float, datetime: datetime = None):
        self.points.append(point)
        old_value = self.value
        if datetime is not None:
            self.datetime = datetime
        if len(self.points) > self.period:
            # This is numpy safe division method.
            self.value = (
                np.divide(
                    (point - self.points[0]),
                    self.points[0],
                    out=np.zeros_like(point),
                    where=self.points[0] != 0,
                )
                * 10000
            )
        # Set the old value and the changed value in base class
        super().update_event_data(old_value)
        # After the update based on a condition we may raise an event.
        self.should_raise_event()

        if self.raise_event:
            event = Event(
                self.type,  # This is the type of the event
                {  # This has the stock id, time frame and the message
                    "stock_id": self.stock_id,
                    "time_frame": self.time_frame,
                    "message": self.message,  # This property is defined in base class
                },
            )
            await self.notify(event)


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


class LinReg(BaseIndicator):
    def __init__(self, stock_id: int, time_frame: int, window_size=5):
        """
        Initialize the streaming statistics calculator.

        Args:
            window_size (int): Number of data points to maintain in rolling window
        """
        super().__init__(stock_id, time_frame)
        self.window_size = window_size
        self.values_y = deque(maxlen=window_size)
        self.values_x = deque(maxlen=window_size)
        # For Nifty we only need independent values as it cannot be compared to another sector.
        if stock_id == 1:
            self.only_independent = True
        else:
            self.only_independent = False

        self.stats = {
            "Independent": {
                "slope": None,
                "y-intercept": None,
                "p-coefficient": None,
                "r-squared": None,
            },
            "Dependent": {
                "slope": None,
                "y-intercept": None,
                "p-coefficient": None,
                "r-squared": None,
            },
        }

    # Here we figure out if we have to raise a event
    def should_raise_event(
        self,
    ):
        # First reset the message and raise event flag and then execute the logic to raise event
        self.message = ""
        self.raise_event = False

    def update(self, new_value_y, new_value_x=None):
        """
        Update with new data point and calculate all statistics.

        Args:
            new_value (float): New data point

        Returns:
            dict: Dictionary containing all calculated statistics
        """
        # Get the PPO for ema 50
        # Add new value to the window

        self.values_y.append(new_value_y)
        # In the case of Nifty we do not have dependent values
        if new_value_x is not None:
            self.values_x.append(new_value_x)

        # Only calculate if we have more than the window size
        if len(self.values_y) < self.window_size:
            return

        # Convert deque to numpy array
        y = np.array(self.values_y)
        x = np.arange(len(y))

        with np.errstate(divide="ignore", invalid="ignore"):
            # Calculate linear regression using numpy's polyfit
            slope, intercept = np.polyfit(x, y, 1)

            # Calculate Pearson's correlation coefficient
            p_coefficient = np.corrcoef(x, y)[0, 1]

            # Calculate R-squared
            y_pred = slope * x + intercept
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)

        if ss_tot == 0:
            r_squared = 0
        else:
            r_squared = 1 - (ss_res / ss_tot)

        self.stats["Independent"]["slope"] = slope
        self.stats["Independent"]["y-intercept"] = intercept
        self.stats["Independent"]["p-coefficient"] = p_coefficient
        self.stats["Independent"]["r-squared"] = r_squared

        # If we want the sector or Nifty independent values.
        if new_value_x is not None:
            x = np.array(self.values_x)

            with np.errstate(divide="ignore", invalid="ignore"):
                # Calculate correlation and regression
                slope, intercept = np.polyfit(x, y, 1)
                p_coefficient = np.corrcoef(x, y)[0, 1]

                # Calculate R-squared
                y_pred = slope * x + intercept
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                ss_res = np.sum((y - y_pred) ** 2)
            if ss_tot == 0:
                r_squared = 0
            else:
                r_squared = 1 - (ss_res / ss_tot)

            self.stats["Dependent"]["slope"] = slope
            self.stats["Dependent"]["y-intercept"] = intercept
            self.stats["Dependent"]["p-coefficient"] = p_coefficient
            self.stats["Dependent"]["r-squared"] = r_squared

    def get_current_values(self):
        """Return current values in the window"""
        return list(self.values_y)


class Vector_Indicators:

    def __init__(
        self,
        time_frame: int,
        indicator_data: Dict[int, Dict[int, Dict[int, BaseIndicator]]],
    ):
        # Load the template with all the stock id, time frame and indicator id
        self.indicator_data: Dict[int, Dict[int, Dict[int, BaseIndicator]]] = (
            indicator_data
        )

        # The column values of the dataframe. This is used during update of technical indicators.
        self.supertrend_first_period_value = "st_fpvalue"
        self.supertrend_first_period_direction = "st_fpdirection"
        self.supertrend_second_period_value = "st_spvalue"
        self.supertrend_second_period_direction = "st_spdirection"
        self.EMA9_column_name = "ema_9"
        self.EMA21_column_name = "ema_21"
        self.ema50_column_name = "ema_50"
        self.ppo_hlc3_ema9_column_name = "ppo_hlc3ema9"
        self.ppo_hlc3ema21_column_name = "ppo_hlc3ema21"
        self.ppo_hlc3st_fp_column_name = "ppo_hlc3st_fp"
        self.ppo_hlc3st_sp_column_name = "ppo_hlc3st_sp"
        self.ind_slope_column_name = "ind_slope"
        self.ind_y_intercept_column_name = "ind_y-intercept"
        self.ind_r_squared_column_name = "ind_r-squared"
        self.ind_p_coef_column_name = "ind_p-coef"
        self.dep_slope_column_name = "dep_slope"
        self.dep_y_intercept_column_name = "dep_y-intercept"
        self.dep_r_squared_column_name = "dep_r-squared"
        self.dep_p_coef_column_name = "dep_p-coef"

        # The indicator id so that the dictionary remains lightweight.
        self.ema9 = 1
        self.ema21 = 2
        self.ema50 = 3
        self.st_fp = 4
        self.st_sp = 5
        self.Lin_Reg = 6
        self.swing_levels = 7

        self.datetime = None

        self.time_frame = time_frame
        self.supertrend_period_length = 10
        if self.time_frame == 15:
            self.supertrend_first_period_multiplier = 1.5
            self.supertrend_second_period_multiplier = 2.5
        else:
            self.supertrend_first_period_multiplier = 2
            self.supertrend_second_period_multiplier = 3
        self.df = self.create_empty_df()

        # Create fresh copy of Indicator class and add them to indicator_data
        self._initialize()

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
            ],
            dtype="float64",
        )

    # Load fresh copy of Indicator classes as we do not have previous days computed indicator classes.
    def _initialize(self):
        # Loop through each stock and assign the indicator instance for stock id, time frame.
        for stock_id, stock in self.indicator_data.items():
            # We do not have intraday data for this stock hence need to comment
            if stock_id == 23 or stock_id == 64 or stock_id == 82:
                continue

            ema9 = EMA(stock_id, self.time_frame, 9)
            self.add_indicator(
                self.ema9,
                ema9,
            )

            ema21 = EMA(stock_id, self.time_frame, 21)
            self.add_indicator(
                self.ema21,
                ema21,
            )

            ema50 = EMA(stock_id, self.time_frame, 50)
            self.add_indicator(
                self.ema50,
                ema50,
            )

            st_fp = SuperTrend(
                stock_id,
                self.time_frame,
                self.supertrend_period_length,
                self.supertrend_first_period_multiplier,
            )
            self.add_indicator(
                self.st_fp,
                st_fp,
            )

            st_sp = SuperTrend(
                stock_id,
                self.time_frame,
                self.supertrend_period_length,
                self.supertrend_second_period_multiplier,
            )
            self.add_indicator(
                self.st_sp,
                st_sp,
            )

            # Based on time frame the rolling window changes. For linear regression we update in 1 min time frame.
            if self.time_frame == 3:
                rolling_window = 9
            elif self.time_frame == 5:
                rolling_window = 15
            elif self.time_frame == 15:
                rolling_window = 45

            lin_reg = LinReg(
                stock_id,
                self.time_frame,
                window_size=rolling_window,
            )
            self.add_indicator(
                self.Lin_Reg,
                lin_reg,
            )

            swing_levels = SwingLevel(stock_id, self.time_frame)
            self.add_indicator(
                self.swing_levels,
                swing_levels,
            )

    def add_indicator(
        self,
        indicator_id: int,
        indicator: BaseIndicator,
    ):
        stock_id: int = indicator.stock_id
        time_frame: int = indicator.time_frame
        self.indicator_data[stock_id][time_frame][indicator_id] = indicator
        # This is the main part where we wire the event with the indicator.
        indicator.add_listener(self.handle_event)

    async def handle_event(self, event: Event):
        print(f"Received event: {event.event_type}")
        # The event data is a JSON with stock_id, time_frame and message. Need to process it
        # By mapping the stock symbol with stock_id and string of time frame.
        # When we build FastAPI this would start stream push data and stop streaming.
        print(f"Event data: {event.data}")

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

        if ema21.value is not None and ema21.value != 0.0:
            self.df.at[stock_id, self.ppo_hlc3ema21_column_name] = round(
                PPO.compute(hlc3, ema21.value) or 0, 2
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
        lin_reg: LinReg = self.indicator_data[stock_id][self.time_frame][self.Lin_Reg]

        if lin_reg.stats["Independent"]["slope"] is not None:
            self.df.at[stock_id, self.ind_slope_column_name] = round(
                lin_reg.stats["Independent"]["slope"] or 0, 2
            )

        if lin_reg.stats["Independent"]["y-intercept"] is not None:
            self.df.at[stock_id, self.ind_y_intercept_column_name] = round(
                lin_reg.stats["Independent"]["y-intercept"] or 0, 2
            )

        if lin_reg.stats["Independent"]["p-coefficient"] is not None:
            self.df.at[stock_id, self.ind_p_coef_column_name] = round(
                lin_reg.stats["Independent"]["p-coefficient"] or 0, 2
            )

        if lin_reg.stats["Independent"]["r-squared"] is not None:
            self.df.at[stock_id, self.ind_r_squared_column_name] = round(
                lin_reg.stats["Independent"]["r-squared"] or 0, 2
            )

        if lin_reg.stats["Dependent"]["slope"] is not None:
            self.df.at[stock_id, self.dep_slope_column_name] = round(
                lin_reg.stats["Dependent"]["slope"] or 0, 2
            )

        if lin_reg.stats["Dependent"]["y-intercept"] is not None:
            self.df.at[stock_id, self.dep_y_intercept_column_name] = round(
                lin_reg.stats["Dependent"]["y-intercept"] or 0, 2
            )

        if lin_reg.stats["Dependent"]["p-coefficient"] is not None:
            self.df.at[stock_id, self.dep_p_coef_column_name] = round(
                lin_reg.stats["Dependent"]["p-coefficient"] or 0, 2
            )

        if lin_reg.stats["Dependent"]["r-squared"] is not None:
            self.df.at[stock_id, self.dep_r_squared_column_name] = round(
                lin_reg.stats["Dependent"]["r-squared"] or 0, 2
            )

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

    def update_stats_values(self, stock_id, new_value_y, new_value_x, sector_id):
        stock_id = int(stock_id)
        # For these stocks we do not have consistent data
        if stock_id == 23 or stock_id == 64 or stock_id == 82:
            return

        # When the Vector Indicators is initialized then self.df is empty and it would throw an error.
        # Hence we do this check.
        if stock_id in self.df.index and sector_id in self.df.index:
            ema_50_y = self.df.at[stock_id, self.ema50_column_name]
            ema_50_x = self.df.at[sector_id, self.ema50_column_name]
        else:
            return

        if ema_50_x is None or ema_50_y is None or ema_50_x == 0.0 or ema_50_y == 0.0:
            return

        # Normalize the values so that we can compare Linear Regression Independent values
        value_y = PPO.compute(new_value_y, ema_50_y)

        lin_reg: LinReg = self.indicator_data[stock_id][self.time_frame][self.Lin_Reg]
        # For nifty we only have independent values
        if stock_id == 1:
            lin_reg.update(value_y)
        else:
            # Normalize the values so that we can compare Linear Regression Dependent values
            value_x = PPO.compute(new_value_x, ema_50_x)
            lin_reg.update(value_y, value_x)

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
            sector_id = df_stock_master.loc[stock_id]["sector_id"]
            stock_hlc3 = row["hlc3"]
            if sector_id not in df_y.index:
                continue
            sector_hlc3 = df_y.loc[sector_id]["hlc3"]
            # logging.info(
            #     f"The sector id: {sector_id} for the stock id: {stock_id}, sector hlc3 value is {sector_hlc3}, stock hlc3 value is {stock_hlc3}"
            # )
            self.update_stats_values(
                stock_id,
                stock_hlc3,
                # Here the sector id is also the stock id and helps fetch the hlc3 value of the sector.
                sector_hlc3,
                sector_id,
            )

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


# # Example usage
# async def main():
#     vi = Vector_Indicators()

#     # Add indicators for multiple stocks and time frames
#     for stock_id in range(2):  # Reduced for brevity
#         for time_frame in [3, 5, 15]:  # time frame array
#             vi.add_indicator(
#                 stock_id,
#                 time_frame,
#                 vi.ema9,
#                 EMA(f"STOCK_{stock_id}", time_frame),
#             )

#     # Simulate updating an indicator
#     await vi.update_indicator("STOCK_0", "1min", "EMA9", {"close": 100})

#     # Pickle the Vector_Indicators instance
#     with open("vector_indicators.pickle", "wb") as f:
#         pickle.dump(vi, f)

#     # Unpickle the Vector_Indicators instance
#     with open("vector_indicators.pickle", "rb") as f:
#         loaded_vi = pickle.load(f)

#     # Test if the event system still works after unpickling
#     stock_id = 1
#     time_frame = 3
#     class_name = "EMA9"
#     await loaded_vi.update_indicator(stock_id, time_frame, class_name, {"close": 110})


# if __name__ == "__main__":
#     asyncio.run(main())
