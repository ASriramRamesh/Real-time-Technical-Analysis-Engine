import numpy as np
import pandas as pd
from collections import deque


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

    def compute(self, point: float):
        points = (list(self.points) + [float(point)])[-self.period :]
        if len(points) == self.period:
            return np.mean(self.points)  # Simple SMA
        elif len(points) > self.period:
            return (point * self.mult) + (self.value * (1 - self.mult))
        return None

    def update(self, point: float):
        self.points.append(point)
        if len(self.points) == self.period:
            self.value = np.mean(self.points)  # Simple SMA
        elif len(self.points) > self.period:
            self.value = (point * self.mult) + (self.value * (1 - self.mult))
        return self.value


class WMA:
    """Weighted Moving Average"""

    def __init__(self, period: int):
        self.period = period
        self.points = deque(maxlen=period)
        self._den = (period * (period + 1)) // 2
        self._weights = np.arange(1, period + 1)
        self.value = None

    def compute(self, point: float):
        points = (list(self.points) + [float(point)])[-self.period :]
        if len(points) == self.period:
            return sum(self._weights * points) / self._den
        return None

    def update(self, point: float):
        self.points.append(point)
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

    def compute(self, point: float):
        return self.ema.compute(point)

    def update(self, point: float):
        self.value = self.ema.update(point)
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

    def update(self, point: float):
        self.points.append(point)
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
                rs = self.avg_gain / self.avg_loss
                self.rsi = 100 - (100 / (1 + rs))
                self.value = self.rsi
        return self.value


class TRANGE:
    """True Range"""

    def __init__(self):
        self.prev_close = None
        self.value = None

    def compute(self, candle):
        if self.prev_close is None:
            return candle["high"] - candle["low"]
        else:
            return max(
                candle["high"] - candle["low"],
                abs(candle["high"] - self.prev_close),
                abs(candle["low"] - self.prev_close),
            )

    def update(self, candle):
        self.value = self.compute(candle)
        self.prev_close = candle["close"]
        return self.value


class ATR:
    """Average True Range"""

    def __init__(self, period, candles=None):
        self.period = period
        self.period_1 = period - 1
        self.TR = TRANGE()
        if candles is None:
            self.atr = 0  # initialised to 0, because values are added to it
            self.value = None
            self.count = 0
        else:
            from talib import ATR

            ta_atr = ATR(candles["high"], candles["low"], candles["close"], period)
            if pd.notna(ta_atr.iloc[-1]):
                self.atr = ta_atr.iloc[-1]
                self.value = self.atr
            else:
                self.atr = 0
                self.value = None
            self.count = len(candles)
            self.TR.update(candles.iloc[-1])

    def compute(self, candle):
        tr = self.TR.compute(candle)
        if self.count < self.period:
            return None
        elif self.count == self.period:
            return (self.atr + tr) / self.period
        else:
            return (self.atr * self.period_1 + tr) / self.period

    def update(self, candle):
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
        else:
            self.ATR = ATR(
                atr_length, candles=candles
            )  # TODO: ATR is getting computed twice
            # Adapted from pandas_ta supertrend.py
            # https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/overlap/supertrend.py
            from talib import ATR as talib_ATR

            _open = candles["open"]
            _high = candles["high"]
            _low = candles["low"]
            _close = candles["close"]
            _median = 0.5 * (_high + _low)  # hl2
            _fatr = factor * talib_ATR(_high, _low, _close, atr_length)
            _basic_upperband = _median + _fatr
            _basic_lowerband = _median - _fatr
            self.lower_band = _basic_lowerband.iloc[0]
            self.upper_band = _basic_upperband.iloc[0]
            for i in range(1, len(candles)):
                if self.super_trend == 1:
                    self.upper_band = _basic_upperband.iloc[i]
                    self.lower_band = max(_basic_lowerband.iloc[i], self.lower_band)
                    if _close.iloc[i] <= self.lower_band:
                        self.super_trend = -1
                else:
                    self.lower_band = _basic_lowerband.iloc[i]
                    self.upper_band = min(_basic_upperband.iloc[i], self.upper_band)
                    if _close.iloc[i] >= self.upper_band:
                        self.super_trend = 1
            if self.super_trend == 1:
                self.final_band = self.lower_band
            else:
                self.final_band = self.upper_band
        self.value = (self.super_trend, self.final_band)  # direction, value

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

    def update(self, candle):
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

    def compute(self, fast_indicator: float, slow_indicator: float):
        if (
            fast_indicator is not None
            and slow_indicator is not None
            and slow_indicator != 0
        ):
            ppo = ((fast_indicator - slow_indicator) / slow_indicator) * 10000
            self.value = ppo

        return self.value

    def update(self, short_ema: float, long_ema: float):
        return self.compute(fast_indicator=short_ema, slow_indicator=long_ema)


class ROC:
    def __init__(self, period: int = 1):
        self.period = period
        self.points = deque(maxlen=period + 1)
        self.value = None

    def compute(self, point: float):
        points = (list(self.points) + [float(point)])[-self.period :]
        if len(points) > self.period:
            return ((point - points[0]) / points[0]) * 10000
        return None

    def update(self, point: float):
        self.points.append(point)
        if len(self.points) > self.period:
            self.value = ((point - self.points[0]) / self.points[0]) * 10000
        return self.value


# roc = ROC(14)  # Period
# for price in prices:
#     roc_value = roc.update(price)
#     print(f"ROC: {roc_value:.2f}")

# ppo = PPO(12, 26, 9)  # Short period, Long period, Signal period
# for price in prices:
#     ppo_value, signal = ppo.update(price)
#     print(f"PPO: {ppo_value:.2f}, Signal: {signal:.2f}")
