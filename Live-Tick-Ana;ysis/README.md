1. It is a bad idea to keep writing to pickle file and retrieving from pickle file. Better to save it in memory.
2. We will load the indicators instance in market_data[stock_id][time_frame]["indicators"][indicator_name]

1. Completed porting the vectorized streaming indicators where we could use multiple stocks and compute the indicators in one shot
2. We need to create a dataframe stock_id and OHLC and HLC3 values and with all the indicator columns
3. We will fill it with the data from the closing of the previous day.
4. When we have 14 indicators for each time frame. In total around 50 pickle files.
5. We can save the pickle file but we will load it only during start of session.
6. In the indicators_dataframe the stock_id would be the index. While this dataframe would be refreshed at every turn
7. We would have 50 dataframe for each indicator and timeframe.
8. For example we would have 96 rows with stock id and the row count will not change.
9. But for every time interval we would have a column addition. The column would be datetime
10. So whenever we have a event then we can get a time slice of Indicators of stock, their sectors and the stocks which belong to these sectors.
11. Saving indicators in columns is not helpful. We need to save it in its own table with column as datetime and stock id as rows.
12. So for a trading day, time frame and stock id folder we would have 14 csv files with 96 rows.

1. Completed the streaming of indicators and swing levels.
2. Saved different time frame csv files from 1st April 2024 for all stocks.
3. We would have to create a master.pkl file which would have the datetime when pickle file has been saved.
4. It is a bad idea to save pickle file continuously for all days. At the most we can backtrack and generate pickle files for 15 indicators.
5. Doing this for OHLC values does not make sense. We could do it for hlc3
6. We need to create a dataframe inside a dataframe.

1. Completed making dataframe inside a dataframe now need to fill it with technical indicators.