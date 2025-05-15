import pandas as pd
import numpy as np


class EMA:
    """Exponential Moving Average for multiple stocks using vectorized operations"""

    def __init__(self, period: int, smoothing_factor: int = 2):
        self.period = period
        self.smoothing_factor = smoothing_factor
        self.mult = smoothing_factor / (1 + period)
        self.history = pd.DataFrame()
        self.values = pd.Series(dtype=float)

    def update(self, df: pd.DataFrame):
        ema_column = f"ema{self.period}"

        # Append new data to history
        self.history = pd.concat([self.history, df]).reset_index(drop=True)

        # Keep only the last 'period' rows for each stock
        self.history = (
            self.history.groupby("stock").tail(self.period).reset_index(drop=True)
        )

        # Calculate EMAs
        if len(self.history) >= self.period:
            # For stocks with enough history, calculate new EMAs
            enough_history = self.history.groupby("stock").size() >= self.period
            stocks_to_update = enough_history[enough_history].index

            if not self.values.empty:
                # Use previous EMAs where available
                previous_emas = self.values.reindex(df["stock"])
                mask = previous_emas.notna()
                df.loc[mask, ema_column] = df.loc[
                    mask, "close"
                ] * self.mult + previous_emas[mask] * (1 - self.mult)

            # Calculate EMAs for stocks with enough history but no previous EMA
            new_emas = (
                self.history[self.history["stock"].isin(stocks_to_update)]
                .groupby("stock")["close"]
                .apply(lambda x: x.ewm(span=self.period, adjust=False).mean().iloc[-1])
            )

            # Update the EMA values
            self.values = pd.concat([self.values, new_emas]).groupby(level=0).last()

            # Update the DataFrame with new EMAs
            df.loc[df["stock"].isin(stocks_to_update), ema_column] = df.loc[
                df["stock"].isin(stocks_to_update), "stock"
            ].map(self.values)

        return df


class RSI:
    """Relative Strength Index for multiple stocks using vectorized operations"""

    def __init__(self, period: int):
        self.period = period
        self.history = pd.DataFrame()

    def update(self, df: pd.DataFrame):
        rsi_column = f"rsi{self.period}"

        # Append new data to history
        self.history = pd.concat([self.history, df]).reset_index(drop=True)

        # Keep only the last 'period + 1' rows for each stock
        self.history = (
            self.history.groupby("stock").tail(self.period + 1).reset_index(drop=True)
        )

        # Calculate price changes
        self.history["change"] = self.history.groupby("stock")["close"].diff()

        # Calculate gains and losses
        self.history["gain"] = self.history["change"].clip(lower=0)
        self.history["loss"] = -self.history["change"].clip(upper=0)

        # Calculate RSI
        if len(self.history) >= self.period + 1:
            # For stocks with enough history, calculate new RSI
            enough_history = self.history.groupby("stock").size() >= self.period + 1
            stocks_to_update = enough_history[enough_history].index

            # Calculate average gains and losses
            avg_gains_losses = (
                self.history[self.history["stock"].isin(stocks_to_update)]
                .groupby("stock")
                .agg(
                    {
                        "gain": lambda x: x.ewm(alpha=1 / self.period, adjust=False)
                        .mean()
                        .iloc[-1],
                        "loss": lambda x: x.ewm(alpha=1 / self.period, adjust=False)
                        .mean()
                        .iloc[-1],
                    }
                )
            )

            # Calculate RSI
            rs = avg_gains_losses["gain"] / avg_gains_losses["loss"]
            new_rsi = 100 - (100 / (1 + rs))

            # Update the DataFrame with new RSI values
            df.loc[df["stock"].isin(stocks_to_update), rsi_column] = df.loc[
                df["stock"].isin(stocks_to_update), "stock"
            ].map(new_rsi)

        return df


class PPO:
    """Percentage Price Oscillator for multiple stocks using vectorized operations"""

    def __init__(self, fast_period: int, slow_period: int):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.history = pd.DataFrame()

    def update(self, df: pd.DataFrame):
        ppo_column = f"ppo_{self.fast_period}_{self.slow_period}"
        fast_ema_column = f"ema{self.fast_period}"
        slow_ema_column = f"ema{self.slow_period}"

        # Append new data to history
        self.history = pd.concat([self.history, df]).reset_index(drop=True)

        # Keep only the last 'slow_period' rows for each stock
        self.history = (
            self.history.groupby("stock").tail(self.slow_period).reset_index(drop=True)
        )

        # Ensure we have both fast and slow EMAs
        if (
            fast_ema_column in self.history.columns
            and slow_ema_column in self.history.columns
        ):
            # Calculate PPO
            ppo = (
                (self.history[fast_ema_column] - self.history[slow_ema_column])
                / self.history[slow_ema_column]
            ) * 10000

            # Get the latest PPO values for each stock
            latest_ppo = ppo.groupby(self.history["stock"]).last()

            # Update the input DataFrame with new PPO values
            df[ppo_column] = df["stock"].map(latest_ppo)

        return df


class ROC:
    """Rate of Change for multiple stocks using vectorized operations"""

    def __init__(self, period: int = 1):
        self.period = period
        self.history = pd.DataFrame()

    def update(self, df: pd.DataFrame):
        roc_column = f"roc{self.period}"

        # Append new data to history
        self.history = pd.concat([self.history, df]).reset_index(drop=True)

        # Keep only the last 'period + 1' rows for each stock
        self.history = (
            self.history.groupby("stock").tail(self.period + 1).reset_index(drop=True)
        )

        # Calculate ROC
        if len(self.history) >= self.period + 1:
            # For stocks with enough history, calculate new ROC
            enough_history = self.history.groupby("stock").size() >= self.period + 1
            stocks_to_update = enough_history[enough_history].index

            # Calculate ROC
            roc = self.history.groupby("stock").apply(lambda x: self._calculate_roc(x))

            # Update the DataFrame with new ROC values
            df.loc[df["stock"].isin(stocks_to_update), roc_column] = df.loc[
                df["stock"].isin(stocks_to_update), "stock"
            ].map(roc)

        return df

    def _calculate_roc(self, group):
        if len(group) >= self.period + 1:
            return (
                (group["close"].iloc[-1] - group["close"].iloc[0])
                / group["close"].iloc[0]
            ) * 10000
        else:
            return np.nan


class ATR:
    """Average True Range for multiple stocks using vectorized operations"""

    def __init__(self, period: int):
        self.period = period
        self.history = pd.DataFrame()
        self.trange_calculator = TRANGE()

    def update(self, df: pd.DataFrame):
        atr_column = f"atr{self.period}"

        # Calculate TRANGE first
        df = self.trange_calculator.update(df)

        # Append new data to history
        self.history = pd.concat([self.history, df]).reset_index(drop=True)

        # Keep only the necessary history for each stock
        self.history = (
            self.history.groupby("stock").tail(self.period).reset_index(drop=True)
        )

        # Ensure we have the 'trange' column
        if "trange" in self.history.columns:
            # Calculate ATR
            self.history["cumulative_tr"] = self.history.groupby("stock")[
                "trange"
            ].cumsum()
            self.history["count"] = self.history.groupby("stock").cumcount() + 1

            # Calculate initial ATR (simple average for the first 'period' points)
            self.history.loc[self.history["count"] == self.period, atr_column] = (
                self.history.loc[self.history["count"] == self.period, "cumulative_tr"]
                / self.period
            )

            # Calculate subsequent ATR values
            self.history.loc[self.history["count"] > self.period, atr_column] = (
                self.history.groupby("stock")[atr_column].shift(1) * (self.period - 1)
                + self.history["trange"]
            ) / self.period

            # Get the latest ATR values for each stock
            latest_atr = self.history.groupby("stock")[atr_column].last()

            # Update the input DataFrame with new ATR values
            df[atr_column] = df["stock"].map(latest_atr)

        return df


class TRANGE:
    """True Range for multiple stocks using vectorized operations"""

    def __init__(self):
        self.history = pd.DataFrame()

    def update(self, df: pd.DataFrame):
        trange_column = "trange"

        # Append new data to history
        self.history = pd.concat([self.history, df]).reset_index(drop=True)

        # Keep only the last 2 rows for each stock (we need current and previous)
        self.history = self.history.groupby("stock").tail(2).reset_index(drop=True)

        # Ensure we have 'high', 'low', and 'close' columns
        required_columns = ["high", "low", "close"]
        if all(col in self.history.columns for col in required_columns):
            # Calculate True Range
            self.history["prev_close"] = self.history.groupby("stock")["close"].shift(1)

            self.history["high_low"] = self.history["high"] - self.history["low"]
            self.history["high_prev_close"] = abs(
                self.history["high"] - self.history["prev_close"]
            )
            self.history["low_prev_close"] = abs(
                self.history["low"] - self.history["prev_close"]
            )

            self.history[trange_column] = self.history[
                ["high_low", "high_prev_close", "low_prev_close"]
            ].max(axis=1)

            # For the first row of each stock (where prev_close is NaN), use high - low
            mask = self.history["prev_close"].isna()
            self.history.loc[mask, trange_column] = self.history.loc[mask, "high_low"]

            # Get the latest TRANGE values for each stock
            latest_trange = self.history.groupby("stock")[trange_column].last()

            # Update the input DataFrame with new TRANGE values
            df[trange_column] = df["stock"].map(latest_trange)

        return df


class SuperTrend:
    """SuperTrend indicator for multiple stocks using vectorized operations"""

    def __init__(self, atr_length: int, factor: float):
        self.atr_length = atr_length
        self.factor = factor
        self.ATR = ATR(atr_length)
        self.history = pd.DataFrame()
        self.values = pd.DataFrame(columns=["super_trend", "final_band"])

    def update(self, df: pd.DataFrame):
        # Append new data to history
        self.history = pd.concat([self.history, df]).reset_index(drop=True)

        # Keep only the last 'atr_length + 1' rows for each stock
        self.history = (
            self.history.groupby("stock")
            .tail(self.atr_length + 1)
            .reset_index(drop=True)
        )

        # Calculate ATR
        self.history = self.ATR.update(self.history)

        # Calculate median price
        self.history["median"] = (self.history["high"] + self.history["low"]) / 2

        # Calculate basic upper and lower bands
        self.history["basic_upper"] = (
            self.history["median"] + self.factor * self.history["atr"]
        )
        self.history["basic_lower"] = (
            self.history["median"] - self.factor * self.history["atr"]
        )

        # Initialize upper and lower bands
        if "upper_band" not in self.history.columns:
            self.history["upper_band"] = self.history["basic_upper"]
            self.history["lower_band"] = self.history["basic_lower"]
            self.history["super_trend"] = 1
            self.history["final_band"] = self.history["lower_band"]

        # Update upper and lower bands
        for stock in self.history["stock"].unique():
            stock_data = self.history[self.history["stock"] == stock]
            for i in range(1, len(stock_data)):
                if stock_data.iloc[i - 1]["super_trend"] == 1:
                    self.history.loc[stock_data.index[i], "lower_band"] = max(
                        stock_data.iloc[i]["basic_lower"],
                        stock_data.iloc[i - 1]["lower_band"],
                    )
                    self.history.loc[stock_data.index[i], "upper_band"] = (
                        stock_data.iloc[i]["basic_upper"]
                    )
                else:
                    self.history.loc[stock_data.index[i], "upper_band"] = min(
                        stock_data.iloc[i]["basic_upper"],
                        stock_data.iloc[i - 1]["upper_band"],
                    )
                    self.history.loc[stock_data.index[i], "lower_band"] = (
                        stock_data.iloc[i]["basic_lower"]
                    )

                if (
                    stock_data.iloc[i - 1]["super_trend"] == 1
                    and stock_data.iloc[i]["close"]
                    <= self.history.loc[stock_data.index[i], "lower_band"]
                ):
                    self.history.loc[stock_data.index[i], "super_trend"] = -1
                elif (
                    stock_data.iloc[i - 1]["super_trend"] == -1
                    and stock_data.iloc[i]["close"]
                    >= self.history.loc[stock_data.index[i], "upper_band"]
                ):
                    self.history.loc[stock_data.index[i], "super_trend"] = 1

                if self.history.loc[stock_data.index[i], "super_trend"] == 1:
                    self.history.loc[stock_data.index[i], "final_band"] = (
                        self.history.loc[stock_data.index[i], "lower_band"]
                    )
                else:
                    self.history.loc[stock_data.index[i], "final_band"] = (
                        self.history.loc[stock_data.index[i], "upper_band"]
                    )

        # Get the latest SuperTrend values for each stock
        latest_values = self.history.groupby("stock")[
            ["super_trend", "final_band"]
        ].last()

        # Update the values DataFrame
        self.values = latest_values

        # Update the input DataFrame with new SuperTrend values
        df["super_trend"] = df["stock"].map(self.values["super_trend"])
        df["final_band"] = df["stock"].map(self.values["final_band"])

        return df


# # Example usage:
# atr14 = ATR(period=14)

# # Simulating data updates
# def update_atr(df):
#     return atr14.update(df)

# # Example DataFrame
# df = pd.DataFrame({
#     'stock': ['AAPL', 'NVD'],
#     'high': [122, 189],
#     'low': [119, 185],
#     'close': [120, 187],
# })

# # Update ATR
# updated_df = update_atr(df)
# print(updated_df)

# # Simulate multiple updates to accumulate enough data for ATR calculation
# for _ in range(15):  # 15 updates to ensure we have enough data for the 14-period ATR
#     new_data = pd.DataFrame({
#         'stock': ['AAPL', 'NVD'],
#         'high': np.random.randint(120, 125, 2),
#         'low': np.random.randint(115, 120, 2),
#         'close': np.random.randint(118, 123, 2),
#     })
#     updated_df = update_atr(new_data)

# print(updated_df)

# # Example usage:
# trange_calculator = TRANGE()

# # Simulating data updates
# def update_trange(df):
#     return trange_calculator.update(df)

# # Example DataFrame
# df = pd.DataFrame({
#     'stock': ['AAPL', 'NVD'],
#     'high': [122, 189],
#     'low': [119, 185],
#     'close': [120, 187],
#     'trange': [0, 0]  # Initial values, will be updated
# })

# # Update TRANGE
# updated_df = update_trange(df)
# print(updated_df)

# # Simulate another update
# df2 = pd.DataFrame({
#     'stock': ['AAPL', 'NVD'],
#     'high': [124, 191],
#     'low': [121, 186],
#     'close': [123, 190],
#     'trange': [0, 0]  # Values from previous update, will be updated again
# })

# updated_df2 = update_trange(df2)
# print(updated_df2)

# # Example usage:
# roc10 = ROC(period=10)

# # Simulating data updates
# def update_roc(df):
#     return roc10.update(df)

# # Example DataFrame
# df = pd.DataFrame({
#     'stock': ['AAPL', 'NVD'],
#     'close': [120, 187],
#     'roc10': [0, 0]  # Initial values, will be updated
# })

# # Update ROC
# updated_df = update_roc(df)
# print(updated_df)

# # Simulate another update with more data points
# df2 = pd.DataFrame({
#     'stock': ['AAPL', 'NVD', 'AAPL', 'NVD', 'AAPL', 'NVD', 'AAPL', 'NVD', 'AAPL', 'NVD', 'AAPL', 'NVD'],
#     'close': [122, 185, 123, 186, 121, 188, 124, 189, 125, 187, 126, 190],
#     'roc10': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Values will be updated
# })

# updated_df2 = update_roc(df2)
# print(updated_df2)

# # Example usage:
# ppo_12_26 = PPO(fast_period=12, slow_period=26)


# # Simulating data updates
# def update_ppo(df):
#     return ppo_12_26.update(df)


# # Example DataFrame
# df = pd.DataFrame(
#     {
#         "stock": ["AAPL", "NVD"],
#         "close": [120, 187],
#         "ema12": [118, 185],
#         "ema26": [115, 180],
#         "ppo_12_26": [0, 0],  # Initial values, will be updated
#     }
# )

# # Update PPO
# updated_df = update_ppo(df)
# print(updated_df)

# # Simulate another update
# df2 = pd.DataFrame(
#     {
#         "stock": ["AAPL", "NVD"],
#         "close": [122, 185],
#         "ema12": [119, 184],
#         "ema26": [116, 181],
#         "ppo_12_26": [0, 0],  # Values from previous update, will be updated again
#     }
# )

# updated_df2 = update_ppo(df2)
# print(updated_df2)


# # Example usage:
# rsi14 = RSI(period=14)


# # Simulating data updates
# def update_rsi(df):
#     return rsi14.update(df)


# # Example DataFrame
# df = pd.DataFrame({"stock": ["AAPL", "NVD"], "close": [120, 187], "rsi14": [70, 40]})

# # Update RSI
# updated_df = update_rsi(df)
# print(updated_df)

# # Simulate another update
# df2 = pd.DataFrame({"stock": ["AAPL", "NVD"], "close": [122, 185], "rsi14": [71, 39]})

# updated_df2 = update_rsi(df2)
# print(updated_df2)

# # Example usage:
# ema9 = EMA(period=9)
# ema21 = EMA(period=21)


# # Simulating data updates every 5 minutes
# def update_emas(df):
#     df = ema9.update(df)
#     df = ema21.update(df)
#     return df


# # Example DataFrame
# df = pd.DataFrame(
#     {
#         "stock": ["AAPL", "NVD"],
#         "close": [120, 187],
#         "ema9": [115, 201],
#         "ema21": [110, 194],
#     }
# )

# # Update EMAs
# updated_df = update_emas(df)
# print(updated_df)

# # Simulate another update
# df2 = pd.DataFrame(
#     {
#         "stock": ["AAPL", "NVD"],
#         "close": [122, 185],
#         "ema9": [115.5, 200],
#         "ema21": [110.5, 193.5],
#     }
# )

# updated_df2 = update_emas(df2)
# print(updated_df2)
