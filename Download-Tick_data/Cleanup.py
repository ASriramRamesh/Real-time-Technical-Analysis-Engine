import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import gc


def generate_ohlc_data(start_date, num_days, num_rows_per_day):
    dates = [
        start_date + timedelta(minutes=5 * i)
        for i in range(num_days * num_rows_per_day)
    ]

    open_prices = np.random.uniform(100, 200, len(dates))
    high_prices = open_prices + np.random.uniform(0, 10, len(dates))
    low_prices = open_prices - np.random.uniform(0, 10, len(dates))
    close_prices = np.random.uniform(low_prices, high_prices)

    df = pd.DataFrame(
        {
            "datetime": dates,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
        }
    )

    return df


# Generate stock data
start_date = datetime(2024, 1, 1, 9, 30)  # Starting from Jan 1, 2024 at 9:30 AM
num_days = 1  # Generate data for 1 day
num_rows_per_day = 200  # 200 rows per day (5-minute intervals for about 16 hours)

stock_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "FB"]
stock_data = {}

for symbol in stock_symbols:
    stock_data[symbol] = generate_ohlc_data(start_date, num_days, num_rows_per_day)

# Example of accessing and displaying data
print("First 5 rows of AAPL data:")
print(stock_data["AAPL"].head())

print("\nShape of each DataFrame:")
for symbol, df in stock_data.items():
    print(f"{symbol}: {df.shape}")

# Example of clearing a specific DataFrame
print("\nClearing AAPL DataFrame:")
stock_data["AAPL"] = stock_data["AAPL"].iloc[0:0]
print(f"AAPL DataFrame shape after clearing: {stock_data['AAPL'].shape}")

# Example of deleting a DataFrame from the dictionary
print("\nDeleting GOOGL DataFrame:")
del stock_data["GOOGL"]
print(f"Remaining keys in stock_data: {list(stock_data.keys())}")

# To clear all DataFrames and delete the dictionary:
for key in list(stock_data.keys()):
    stock_data[key] = stock_data[key].iloc[0:0]
    del stock_data[key]
stock_data.clear()
del stock_data

gc.collect()
