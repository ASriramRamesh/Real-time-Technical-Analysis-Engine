import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


class LiveMarketLevels:
    """
    A class to dynamically track support and resistance levels in a live market scenario.

    Attributes:
    - max_lookback: Maximum number of recent data points to consider
    - support_levels: List of unique support levels
    - resistance_levels: List of unique resistance levels
    - df_full: Full DataFrame of market data
    - df_support: DataFrame of current support levels
    - df_resistance: DataFrame of current resistance levels
    """

    def __init__(self, max_lookback: int = 200, price_range: int = 5):
        """
        Initialize the LiveMarketLevels tracker.

        :param max_lookback: Maximum number of recent data points to maintain
        """
        # Initialize DataFrames with expected columns
        self.df_full = pd.DataFrame(columns=["time_string", "trend", "move", "level"])
        self.df_support = pd.DataFrame(columns=["time_string", "level"])
        self.df_resistance = pd.DataFrame(columns=["time_string", "level"])

        # Track support and resistance levels
        self.support_levels: List[float] = []
        self.resistance_levels: List[float] = []

        # Maximum number of historical data points to keep
        self.max_lookback = max_lookback
        self.price_range = price_range

    def _is_unique_support_level(self, level: float) -> bool:
        """
        Check if the support level is unique (not within ±5 of existing levels).

        :param level: Price level to check
        :return: True if unique, False otherwise
        """
        return not any(
            abs(level - existing) <= self.price_range
            for existing in self.support_levels
        )

    def _is_unique_resistance_level(self, level: float) -> bool:
        """
        Check if the resistance level is unique (not within ±5 of existing levels).

        :param level: Price level to check
        :return: True if unique, False otherwise
        """
        return not any(
            abs(level - existing) <= self.price_range
            for existing in self.resistance_levels
        )

    def update(
        self, time_string: str, trend: int, move: int, level: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Update market data and recalculate support and resistance levels.

        :param time_string: Time of the data point
        :param trend: Trend direction (1 for up, -1 for down)
        :param move: Move type (1 for continuation, -1 for pullback)
        :param level: Normalized price level
        :return: Updated support and resistance DataFrames
        """
        # Add new data point to full DataFrame
        new_row = pd.DataFrame(
            {
                "time_string": [time_string],
                "trend": [trend],
                "move": [move],
                "level": [level],
            }
        )
        self.df_full = pd.concat([self.df_full, new_row]).tail(self.max_lookback)

        # Recalculate support levels
        self.support_levels.clear()
        support_times = []
        support_candidates = self.df_full[
            (self.df_full["trend"] == -1) & (self.df_full["move"] == 1)
        ]

        for _, row in support_candidates.iterrows():
            if self._is_unique_support_level(row["level"]):
                self.support_levels.append(row["level"])
                support_times.append(row["time_string"])

        # Update support DataFrame
        self.df_support = pd.DataFrame(
            {"time_string": support_times, "level": self.support_levels}
        )

        # Recalculate resistance levels
        self.resistance_levels.clear()
        resistance_times = []
        resistance_candidates = self.df_full[
            (self.df_full["trend"] == 1) & (self.df_full["move"] == -1)
        ]

        for _, row in resistance_candidates.iterrows():
            if self._is_unique_resistance_level(row["level"]):
                self.resistance_levels.append(row["level"])
                resistance_times.append(row["time_string"])

        # Update resistance DataFrame
        self.df_resistance = pd.DataFrame(
            {"time_string": resistance_times, "level": self.resistance_levels}
        )

        return self.df_support, self.df_resistance

    def get_current_levels(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get current support and resistance levels.

        :return: Current support and resistance DataFrames
        """
        return self.df_support, self.df_resistance


# Example usage demonstration
def simulate_live_market():
    """
    Simulate a live market scenario with continuous data updates.
    """
    # Initialize the LiveMarketLevels tracker
    market_tracker = LiveMarketLevels(max_lookback=50, price_range=5)

    # Sample market data (simulating live updates)
    market_data = [
        ("15:06", -1, 1, -31),
        ("14:57", -1, -1, -4),
        ("14:48", 1, 1, -17),
        ("14:42", 1, -1, -8),
        # ... more data points can be added
    ]

    # Simulate live market updates
    for time_string, trend, move, level in market_data:
        df_support, df_resistance = market_tracker.update(
            time_string, trend, move, level
        )

        print(f"\nUpdate at {time_string}")
        print("Support Levels:")
        print(df_support)
        print("\nResistance Levels:")
        print(df_resistance)

    return market_tracker


def process_stock_price_swings(df):
    """
    Remove specific records based on the given conditions:
    1. Remove records with is_up_trend = 1 and is_up_move = -1
    if preceded by a record with is_up_trend = 1 and is_up_move = 1
    2. Remove records with is_up_trend = -1 and is_up_move = 1
    if preceded by a record with is_up_trend = -1 and is_up_move = -1

    Args:
    df (pandas.DataFrame): Input dataframe with columns 'time', 'is_up_trend', 'is_up_move', 'level'

    Returns:
    pandas.DataFrame: Processed dataframe with specified records removed
    """
    # Create a copy of the dataframe to avoid modifying the original
    processed_df = df.copy()

    # Sort the dataframe by time in descending order to process chronologically
    processed_df = processed_df.sort_values("time", ascending=False).reset_index(
        drop=True
    )

    # Identify rows to remove
    rows_to_remove = []
    for i in range(1, len(processed_df)):
        # Check uptrend conditions
        if (
            processed_df.loc[i - 1, "is_up_trend"] == 1
            and processed_df.loc[i - 1, "is_up_move"] == 1
            and processed_df.loc[i, "is_up_trend"] == 1
            and processed_df.loc[i, "is_up_move"] == -1
        ):
            rows_to_remove.append(i)

        # Check downtrend conditions
        if (
            processed_df.loc[i - 1, "is_up_trend"] == -1
            and processed_df.loc[i - 1, "is_up_move"] == -1
            and processed_df.loc[i, "is_up_trend"] == -1
            and processed_df.loc[i, "is_up_move"] == 1
        ):
            rows_to_remove.append(i)

    # Remove identified rows
    processed_df = processed_df.drop(rows_to_remove).reset_index(drop=True)

    # Sort back to original time order
    processed_df = processed_df.sort_values("time", ascending=False).reset_index(
        drop=True
    )

    return processed_df


# Run simulation if script is executed directly
if __name__ == "__main__":
    market_tracker = simulate_live_market()

    # # Use pd.concat to add the new row to the front of the DataFrame
    # df = pd.concat([new_row, df], ignore_index=True)

    # We need to drop rows where the pullback lead to continuation of the trend. This is to reduce support and resistance.
    # If we are computing in live market we need to only do it when we have is_up_trend = 1 and is_up_move = 1
    # or when we have is_up_trend = -1 and is_up_move = -1. If it is continuation of trend we remove the prior record and
    # update from the last update time.
    # Example input data
    data = {
        "time": [
            "15:06",
            "14:57",
            "14:48",
            "14:42",
            "14:30",
            "14:03",
            "13:45",
            "13:39",
            "13:33",
            "13:21",
            "13:09",
            "12:57",
            "12:36",
            "12:30",
            "11:48",
            "11:03",
            "11:00",
            "10:54",
            "10:36",
            "10:21",
            "10:09",
            "09:57",
            "09:33",
            "09:15",
        ],
        "is_up_trend": [
            -1,
            -1,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
        ],
        "is_up_move": [
            1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
        ],
        "level": [
            -31,
            -4,
            -17,
            -8,
            -30,
            -20,
            -43,
            -52,
            -33,
            -27,
            -42,
            -27,
            -46,
            -47,
            -10,
            -5,
            -15,
            -6,
            -31,
            -37,
            3,
            -18,
            29,
            38,
        ],
    }

    df = pd.DataFrame(data)

    # Process the dataframe
    processed_df = process_stock_price_swings(df)

    # Display original and processed dataframes
    print("Original DataFrame:")
    print(df)
    print("\nProcessed DataFrame:")
    print(processed_df)
