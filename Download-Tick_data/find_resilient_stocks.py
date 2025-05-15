import yfinance as yf
import pandas as pd
import talib as ta

# Define the sector stocks you want to analyze
banking_stocks = ['JPM', 'BAC', 'C', 'WFC', 'GS']  # Example banking stocks

# Set parameters for EMA and RSI
ema_period = 9
rsi_period = 14
smoothed_ma_period = 10  # Example for smoothed MA of RSI

def analyze_stock(stock_symbol):
    # Download intraday data for the stock (e.g., 5-minute interval for 1 day)
    data = yf.download(stock_symbol, period="1d", interval="5m")
    
    # Calculate 9 EMA
    data['EMA_9'] = ta.EMA(data['Close'], timeperiod=ema_period)
    
    # Calculate RSI
    data['RSI'] = ta.RSI(data['Close'], timeperiod=rsi_period)
    
    # Calculate Smoothed MA of RSI
    data['Smoothed_RSI'] = ta.SMA(data['RSI'], timeperiod=smoothed_ma_period)
    
    # Identify points where price is above 9 EMA and RSI is above Smoothed RSI
    data['Resilient'] = (data['Close'] > data['EMA_9']) & (data['RSI'] > data['Smoothed_RSI'])
    
    return data

# Analyze all stocks in the sector
resilient_stocks = {}

for stock in banking_stocks:
    stock_data = analyze_stock(stock)
    
    # Check if there are any resilient points in the pullback
    if stock_data['Resilient'].any():
        resilient_stocks[stock] = stock_data[stock_data['Resilient']]

# Output the resilient stock(s) during the pullback
if resilient_stocks:
    for stock, data in resilient_stocks.items():
        print(f"\nResilient Stock: {stock}")
        print(data.tail())  # Show last few rows of the resilient points
else:
    print("No resilient stocks found in this sector.")
