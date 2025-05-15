import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
import os
import pickle


class RollingStat:
    """Abstract Class - Used for functions which require computing stat on fixed window queue"""

    def __init__(self, period: int, func, points=None):
        assert period > 1, "Period needs to be greater than 1."
        self.period = period
        if points is None:
            self.points = deque(maxlen=period)
        else:
            self.points = deque(points[-period:], maxlen=period)
        self.func = func

    def compute(self, point: float):
        points = (list(self.points) + [float(point)])[-self.period :]
        if len(points) == self.period:
            return self.func(points)
        return None

    def update(self, point: float):
        self.points.append(float(point))
        return self.value

    @property
    def value(self):
        if len(self.points) == self.period:
            return self.func(self.points)
        return None


class Max(RollingStat):
    """Maximum in a rolling window"""

    def __init__(self, period: int, points=None):
        super().__init__(period=period, func=max, points=points)


class Min(RollingStat):
    """Minimum in a rolling window"""

    def __init__(self, period: int, points=None):
        super().__init__(period=period, func=min, points=points)


class SMA(RollingStat):
    """Simple Moving Average"""

    def __init__(self, period: int, points=None):
        super().__init__(period=period, func=np.mean, points=points)
        # TODO: Any efficient way rather than computing everytime?


class SD(RollingStat):
    """Standard Deviation"""

    def __init__(self, period: int, points=None):
        super().__init__(period=period, func=np.std, points=points)
        # TODO: Any efficient way rather than computing everytime?


class EMA:
    """Exponential Moving Average"""

    def __init__(self, period: int, smoothing_factor: int = 2):
        self.period = period
        self.smoothing_factor = smoothing_factor
        self.mult = smoothing_factor / (1 + period)
        self.points = deque(maxlen=period + 1)
        self.value = None
        self.datetime = None

    def compute(self, point: float):
        points = (list(self.points) + [float(point)])[-self.period :]
        if len(points) == self.period:
            return np.mean(self.points)  # Simple SMA
        elif len(points) > self.period:
            return (point * self.mult) + (self.value * (1 - self.mult))
        return None

    def update(self, point: float, datetime: datetime = None):
        self.points.append(point)
        if datetime is not None:
            self.datetime = datetime
        if len(self.points) == self.period:
            self.value = np.mean(self.points)  # Simple SMA
        elif len(self.points) > self.period:
            self.value = (point * self.mult) + (self.value * (1 - self.mult))
        return self.value

    def save_to_pickle(self, time_frame: str, stock_id: str):
        """Save the EMA instance to a pickle file"""
        folder_path = os.path.join(os.getcwd(), "streaming", time_frame, stock_id)
        # os.makedirs(folder_path, exist_ok=True)

        # Create filename using datetime if available, otherwise use timestamp
        if self.datetime:
            timestamp = self.datetime.strftime("%Y%m%d_%H%M%S")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ema_{self.period}_{timestamp}.pkl"

        filepath = os.path.join(folder_path, filename)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        return filepath

    def save_to_pickle_folder(self, folder_path: str):
        """Save the EMA instance to a pickle file"""
        filename = f"ema_{self.period}.pkl"

        filepath = os.path.join(folder_path, filename)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        return filepath

    @staticmethod
    def load_from_pickle(
        time_frame: str, stock_id: str, period: str, timestamp: str
    ) -> "EMA":
        """Load an EMA instance from a pickle file"""
        folder_path = os.path.join(os.getcwd(), "streaming", time_frame, stock_id)
        filename = f"ema_{period}_{timestamp}.pkl"
        filepath = os.path.join(folder_path, filename)

        with open(filepath, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_from_pickle_folder(folder_path: str, period: str) -> "EMA":
        """Load an EMA instance from a pickle file"""
        filename = f"ema_{period}.pkl"
        filepath = os.path.join(folder_path, filename)

        with open(filepath, "rb") as f:
            return pickle.load(f)


class WMA:
    """Weighted Moving Average"""

    def __init__(self, period: int):
        self.period = period
        self.points = deque(maxlen=period)
        self._den = (period * (period + 1)) // 2
        self._weights = np.arange(1, period + 1)
        self.value = None
        self.datetime = None

    def compute(self, point: float):
        points = (list(self.points) + [float(point)])[-self.period :]
        if len(points) == self.period:
            return sum(self._weights * points) / self._den
        return None

    def update(self, point: float, datetime: datetime = None):
        self.points.append(point)
        if datetime is not None:
            self.datetime = datetime
        if len(self.points) == self.period:
            self.value = sum(self._weights * self.points) / self._den
        return self.value


class SMMA:
    """Smoothed Moving Average"""

    def __init__(self, period: int):
        assert period > 1, "Period needs to be greater than 1."
        self.period = period
        self.ema_period = period * 2 - 1
        # https://stackoverflow.com/a/72533211/6430403
        self.ema = EMA(self.ema_period)
        self.datetime = None

    def compute(self, point: float):
        return self.ema.compute(point)

    def update(self, point: float, datetime: datetime = None):
        self.value = self.ema.update(point)
        if datetime is not None:
            self.datetime = datetime
        return self.value


class RSI:
    """Relative Strength Index"""

    def __init__(self, period: int):
        self.period = period
        self._period_minus_1 = period - 1
        self._period_plus_1 = period + 1
        self.points = deque(maxlen=self._period_plus_1)
        self.losses = deque(maxlen=self._period_plus_1)
        self.gains = deque(maxlen=self._period_plus_1)
        self.avg_gain = None
        self.avg_loss = None
        self.rsi = None
        self.value = None
        self.datetime = None

    def update(self, point: float, datetime: datetime = None):
        self.points.append(point)
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
        return self.value

    def save_to_pickle(self, time_frame: str, stock_id: str):
        """Save the RSI instance to a pickle file"""
        folder_path = os.path.join(os.getcwd(), "streaming", time_frame, stock_id)
        # os.makedirs(folder_path, exist_ok=True)

        # Create filename using datetime if available, otherwise use timestamp
        if self.datetime:
            timestamp = self.datetime.strftime("%Y%m%d_%H%M%S")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rsi_{self.period}_{timestamp}.pkl"

        filepath = os.path.join(folder_path, filename)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        return filepath

    def save_to_pickle_folder(self, folder_path: str):
        """Save the RSI instance to a pickle file"""
        filename = f"rsi_{self.period}.pkl"

        filepath = os.path.join(folder_path, filename)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        return filepath

    @staticmethod
    def load_from_pickle(
        time_frame: str, stock_id: str, period: str, timestamp: str
    ) -> "RSI":
        """Load an EMA instance from a pickle file"""
        folder_path = os.path.join(os.getcwd(), "streaming", time_frame, stock_id)
        filename = f"rsi_{period}_{timestamp}.pkl"
        filepath = os.path.join(folder_path, filename)

        with open(filepath, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_from_pickle_folder(folder_path: str, period: str) -> "RSI":
        """Load an EMA instance from a pickle file"""
        filename = f"rsi_{period}.pkl"
        filepath = os.path.join(folder_path, filename)

        with open(filepath, "rb") as f:
            return pickle.load(f)


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


class SuperTrend:
    def __init__(self, atr_length, factor, candles=None):
        self.factor = factor
        self.super_trend = 1
        if candles is None:
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

    def update(self, candle, datetime: datetime = None):
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
        return self.value

    def save_to_pickle(self, time_frame: str, stock_id: str, period: str):
        """Save the SuperTrend instance to a pickle file"""
        folder_path = os.path.join(os.getcwd(), "streaming", time_frame, stock_id)
        # os.makedirs(folder_path, exist_ok=True)

        # Create filename using datetime if available, otherwise use timestamp
        if self.datetime:
            timestamp = self.datetime.strftime("%Y%m%d_%H%M%S")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"supertrend_{period}_{timestamp}.pkl"

        filepath = os.path.join(folder_path, filename)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        return filepath

    def save_to_pickle_folder(self, folder_path: str, period: str):
        filename = f"supertrend_{period}.pkl"

        filepath = os.path.join(folder_path, filename)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        return filepath

    @staticmethod
    def load_from_pickle(
        time_frame: str, stock_id: str, period: str, timestamp: str
    ) -> "SuperTrend":
        """Load an SuperTrend instance from a pickle file"""
        folder_path = os.path.join(os.getcwd(), "streaming", time_frame, stock_id)
        filename = f"supertrend_{period}_{timestamp}.pkl"
        filepath = os.path.join(folder_path, filename)

        with open(filepath, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_from_pickle_folder(folder_path: str, period: str) -> "SuperTrend":
        """Load an SuperTrend instance from a pickle file"""
        filename = f"supertrend_{period}.pkl"
        filepath = os.path.join(folder_path, filename)

        with open(filepath, "rb") as f:
            return pickle.load(f)


import operator

COMPARATORS = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
}


class IsOrder:
    """
    Checks if a given list of elements is in an order. eg. all increasing
    examples:
    - all_increasing = IsOrder('>')
    - all_decreasing = IsOrder('<=')
    - all_equal = IsOrder('==')
    - doubling = IsOrder(lambda a,b: a == 2*b)

    data = [1, 2, 3, 4, 5, 6]
    data_length = len(data)
    is_increasing = IsOrder('>', data_length)

    expected_output = [False, False, False, False, False, True, False]

    for x in data:
        print(is_increasing.update(x))
    """

    def __init__(self, comparator, length=2):
        self.comparator = COMPARATORS.get(comparator, comparator)
        self.length = length
        self.q = deque(length * [None], maxlen=length)
        self.fresh = True
        self.order_idx = 1
        self.is_ordered = False
        self.value = False

    def update(self, element):
        self.q.append(element)
        if self.fresh:
            self.fresh = False
            return False
        # comparator (new element, old element)
        if self.comparator(element, self.q[-2]):
            self.order_idx += 1
        else:
            self.order_idx = 1
        self.is_ordered = self.order_idx >= self.length
        self.value = self.is_ordered
        return self.value


class PPO:
    """Percentage Price Oscillator: We have to pass both the ema 9, 21 or ema9, hlc3 or rsi, smoothed MA"""

    def __init__(
        self,
    ):
        self.value = None
        self.datetime = None

    def compute(self, fast_indicator: float, slow_indicator: float):
        if (
            fast_indicator is not None
            and slow_indicator is not None
            and slow_indicator != 0
        ):
            ppo = ((fast_indicator - slow_indicator) / slow_indicator) * 10000
            self.value = ppo

        return self.value

    def update(self, short_ema: float, long_ema: float, datetime: datetime = None):
        if datetime is not None:
            self.datetime = datetime
        return self.compute(fast_indicator=short_ema, slow_indicator=long_ema)

    def save_to_pickle_folder(self, folder_path: str, period: str):
        filename = f"ppo_{period}.pkl"

        filepath = os.path.join(folder_path, filename)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        return filepath

    @staticmethod
    def load_from_pickle_folder(folder_path: str, period: str) -> "PPO":
        filename = f"ppo_{period}.pkl"
        filepath = os.path.join(folder_path, filename)

        with open(filepath, "rb") as f:
            return pickle.load(f)


class ROC:
    def __init__(self, period: int = 1):
        self.period = period
        self.points = deque(maxlen=period + 1)
        self.value = None
        self.datetime = None

    def compute(self, point: float):
        points = (list(self.points) + [float(point)])[-self.period :]
        if len(points) > self.period:
            return ((point - points[0]) / points[0]) * 10000
        return None

    def update(self, point: float, datetime: datetime = None):
        self.points.append(point)
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
            # self.value = ((point - self.points[0]) / self.points[0]) * 10000
        return self.value

    def save_to_pickle_folder(self, folder_path: str, period: str):
        filename = f"roc_{period}.pkl"

        filepath = os.path.join(folder_path, filename)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        return filepath

    @staticmethod
    def load_from_pickle_folder(folder_path: str, period: str) -> "ROC":
        filename = f"roc_{period}.pkl"
        filepath = os.path.join(folder_path, filename)

        with open(filepath, "rb") as f:
            return pickle.load(f)


# roc = ROC(14)  # Period
# for price in prices:
#     roc_value = roc.update(price)
#     print(f"ROC: {roc_value:.2f}")

# ppo = PPO(12, 26, 9)  # Short period, Long period, Signal period
# for price in prices:
#     ppo_value, signal = ppo.update(price)
#     print(f"PPO: {ppo_value:.2f}, Signal: {signal:.2f}")


# from ta.vector_indicators import (
#     RSI as vector_rsi,
#     SuperTrend as vector_supertrend,
#     EMA as vector_ema,
#     PPO as vector_ppo,
#     ROC as vector_roc,
# )


# We need to add a messages method or a event that could be subscribed by parent class.
# class Vector_Indicators:
#     """Vector Indicators"""

#     def __init__(
#         self,
#         time_frame,
#         market_data_pickle_file_name=None,
#         market_data_pickle_folder=None,
#     ):
#         self.ema9 = "ema9"
#         self.ema21 = "ema21"
#         self.rsi14 = "rsi14"
#         self.st_fp = "st_fp"
#         self.st_sp = "st_sp"
#         self.smoothed_rsi = "smoothed_rsi"
#         self.ppo_hlc3 = "ppo_hlc3"
#         self.ppo_ema9 = "ppo_ema9"
#         self.ppo_rsi = "ppo_rsi"
#         self.roc_hlc3 = "roc_hlc3"
#         self.roc_ema9 = "roc_ema9"
#         self.roc_ema21 = "roc_ema21"
#         self.roc_rsi = "roc_rsi"
#         self.roc_smoothedrsi = "roc_smoothedrsi"
#         self.swing_levels = "swing_levels"
#         self.datetime = None

#         if market_data_pickle_file_name is None:
#             self.market_data = load_dict_from_file("market_data.pkl")
#         else:
#             self.market_data = load_pickle_from_file(
#                 market_data_pickle_file_name, folder_path=market_data_pickle_folder
#             )

#         self.time_frame = time_frame
#         self.supertrend_period_length = 10
#         if self.time_frame == 3:
#             self.supertrend_first_period_multiplier = 3
#             self.supertrend_second_period_multiplier = 4
#         elif self.time_frame == 5:
#             self.supertrend_first_period_multiplier = 2
#             self.supertrend_second_period_multiplier = 3
#         else:
#             self.supertrend_first_period_multiplier = 1.5
#             self.supertrend_second_period_multiplier = 2.5

#         for stock_id, stock in self.market_data.items():
#             stock_symbol = stock["symbol"]

#             # We do not have intraday data for this stock hence need to comment
#             if stock_symbol == "VBL" or stock_symbol == "IDFC":
#                 continue

#             stock[self.time_frame]["indicators"][self.ema9] = streaming_ema(9)
#             stock[self.time_frame]["indicators"][self.ema21] = streaming_ema(21)
#             stock[self.time_frame]["indicators"][self.rsi14] = streaming_rsi(14)

#             stock[self.time_frame]["indicators"][self.st_fp] = streaming_supertrend(
#                 self.supertrend_period_length, self.supertrend_first_period_multiplier
#             )
#             stock[self.time_frame]["indicators"][self.st_sp] = streaming_supertrend(
#                 self.supertrend_period_length, self.supertrend_second_period_multiplier
#             )

#             stock[self.time_frame]["indicators"][self.smoothed_rsi] = streaming_ema(14)
#             stock[self.time_frame]["indicators"][self.ppo_hlc3] = streaming_ppo()
#             stock[self.time_frame]["indicators"][self.ppo_ema9] = streaming_ppo()
#             stock[self.time_frame]["indicators"][self.ppo_rsi] = streaming_ppo()

#             stock[self.time_frame]["indicators"][self.roc_hlc3] = streaming_roc()
#             stock[self.time_frame]["indicators"][self.roc_ema9] = streaming_roc()
#             stock[self.time_frame]["indicators"][self.roc_ema21] = streaming_roc()
#             stock[self.time_frame]["indicators"][self.roc_rsi] = streaming_roc()
#             stock[self.time_frame]["indicators"][self.roc_smoothedrsi] = streaming_roc()
#             stock[self.time_frame]["indicators"][self.swing_levels] = SwingLevel()

#     def update(self, df, trading_date: datetime = None):
#         self.datetime = trading_date
#         for stock_id, row in df.iterrows():
#             open = row["open"]
#             high = row["high"]
#             low = row["low"]
#             close = row["close"]
#             hlc3 = round((high + low + close) / 3, 2)

#             indicators = self.market_data[stock_id][self.time_frame]["indicators"]
#             # Creating indicator columns in the dataframe
#             df[supertrend_first_period_value] = 0.0
#             df[supertrend_first_period_direction] = 0.0
#             df[supertrend_second_period_value] = 0.0
#             df[supertrend_second_period_direction] = 0.0
#             df[EMA9_column_name] = 0.0
#             df[EMA21_column_name] = 0.0
#             df[RSI_column_name] = 0.0
#             df[Smoothed_RSI_column_name] = 0.0
#             df[ppo_hlc3_ema9_column_name] = 0.0
#             df[ppo_ema9_ema21_column_name] = 0.0
#             df[ppo_rsi_emarsi_column_name] = 0.0
#             df[roc_hlc3_column_name] = 0.0
#             df[roc_ema9_column_name] = 0.0
#             df[roc_ema21_column_name] = 0.0
#             df[roc_rsi_column_name] = 0.0
#             df[roc_smoothed_rsi_column_name] = 0.0

#             ema9 = indicators[self.ema9]
#             ema9_value = ema9.update(hlc3, self.datetime)
#             df.at[stock_id, EMA9_column_name] = np.float64(round(ema9_value or 0, 2))

#             ema21 = indicators[self.ema21]
#             ema21_value = ema21.update(hlc3, self.datetime)
#             df.at[stock_id, EMA21_column_name] = np.float64(round(ema21_value or 0, 2))

#             rsi14 = indicators[self.rsi14]
#             rsi_value = rsi14.update(close, self.datetime)
#             df.at[stock_id, RSI_column_name] = np.float64(round(rsi_value or 0, 2))

#             # Only the first period super trend value is required to get the swing levels.
#             st_fp = indicators[self.st_fp]
#             st_fp_direction, st_fp_value = st_fp.update(row, self.datetime)
#             df.at[stock_id, supertrend_first_period_value] = np.float64(
#                 round(st_fp_value or 0, 2)
#             )
#             df.at[stock_id, supertrend_first_period_direction] = st_fp_direction

#             st_sp = indicators[self.st_sp]
#             st_sp_direction, st_sp_value = st_sp.update(row, self.datetime)
#             df.at[stock_id, supertrend_second_period_value] = np.float64(
#                 round(st_sp_value or 0, 2)
#             )
#             df.at[stock_id, supertrend_second_period_direction] = st_sp_direction

#             if rsi_value is not None:
#                 smoothed_rsi = indicators[self.smoothed_rsi]
#                 smoothed_rsi_value = smoothed_rsi.update(rsi_value, self.datetime)
#                 df.at[stock_id, Smoothed_RSI_column_name] = np.float64(
#                     round(smoothed_rsi_value or 0, 2)
#                 )

#             roc_hlc3 = indicators[self.roc_hlc3]
#             roc_hlc3_value = roc_hlc3.update(hlc3)
#             df.at[stock_id, roc_hlc3_column_name] = np.float64(
#                 round(roc_hlc3_value or 0, 2)
#             )

#             if ema9_value is not None:
#                 roc_ema9 = indicators[self.roc_ema9]
#                 roc_ema9_value = roc_ema9.update(ema9_value)
#                 df.at[stock_id, roc_ema9_column_name] = np.float64(
#                     round(roc_ema9_value or 0, 2)
#                 )
#             if ema21_value is not None:
#                 roc_ema21 = indicators[self.roc_ema21]
#                 roc_ema21_value = roc_ema21.update(ema21_value)
#                 df.at[stock_id, roc_ema21_column_name] = np.float64(
#                     round(roc_ema21_value or 0, 2)
#                 )
#             if rsi_value is not None:
#                 roc_rsi = indicators[self.roc_rsi]
#                 roc_rsi_value = roc_rsi.update(rsi_value)
#                 df.at[stock_id, roc_rsi_column_name] = np.float64(
#                     round(roc_rsi_value or 0, 2)
#                 )

#             if smoothed_rsi_value is not None:
#                 roc_smoothedrsi = indicators[self.roc_smoothedrsi]
#                 roc_smoothedrsi_value = roc_smoothedrsi.update(smoothed_rsi_value)
#                 df.at[stock_id, roc_smoothed_rsi_column_name] = np.float64(
#                     round(roc_smoothedrsi_value or 0, 2)
#                 )
#             if ema9_value is not None:
#                 ppo_hlc3 = indicators[self.ppo_hlc3]
#                 ppo_hlc3_value = ppo_hlc3.update(hlc3, ema9_value)
#                 df.at[stock_id, ppo_hlc3_ema9_column_name] = np.float64(
#                     round(ppo_hlc3_value or 0, 2)
#                 )

#             if ema9_value is not None and ema21_value is not None:
#                 ppo_ema9 = indicators[self.ppo_ema9]
#                 ppo_ema9_value = ppo_ema9.update(ema9_value, ema21_value)
#                 df.at[stock_id, ppo_ema9_ema21_column_name] = np.float64(
#                     round(ppo_ema9_value or 0, 2)
#                 )

#             if smoothed_rsi_value is not None:
#                 ppo_rsi = indicators[self.ppo_rsi]
#                 ppo_rsi_value = ppo_rsi.update(rsi_value, smoothed_rsi_value)
#                 df.at[stock_id, ppo_rsi_emarsi_column_name] = np.float64(
#                     round(ppo_rsi_value or 0, 2)
#                 )

#             # This is different in the sense that it does not populate df but maintains df in the class
#             # We would need to raise the event and bubble it up.
#             swing_levels = indicators[self.swing_levels]
#             # Data row addition happens occasionally. But every function call updates internal state.
#             swing_levels.loop_min_create_data_rows(self.datetime, row)

#         return df
