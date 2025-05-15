import urllib.parse
import pandas as pd
import pytz, os

import requests
import json, requests
from datetime import datetime, timedelta, time

import asyncio
import aiohttp

timeZone = "Asia/Kolkata"
TIME_ZONE = pytz.timezone(timeZone)

# We need to use asyncio to download files and save it in a different folder
# verify all files have been downloaded.
# add the sector for Watchlist_equity
# need to look at more securities from Nifty 100, 200 and 500. After
# We need to seperate the function for Historical data and daily 1 min data.
# We need to set the from date and too date to fetch the data
# We need to try and get the right instrument key for Nifty AUTO, Nifty ENERGY, Nifty INFRA, Nifty METAL, Nifty PHARMA


""" async def fetch(s, url):
    async with s.get(url) as r:
        if r.status != 200:
            r.raise_for_status()
        return r.text()


async def fetch_all(s, urls):
    tasks = []
    for url in urls:
        task = asyncio.create_task(fetch(s, url))
        tasks.append(task)

    res = await asyncio.gather(*tasks)


async def getBunchedJSON():

    async with aiohttp.ClientSession() as session:
        htmls = await fetch_all(session, urls)
        print(htmls)
"""


def getHistoricalData(instrument_key, tradingsymbol):
    res = None
    try:
        parseInstrument = urllib.parse.quote(instrument_key)
        timeFrame = "1minute"
        # Set the days variable for how many days data is required.
        # fromDate = (datetime.now(TIME_ZONE) - timedelta(days=1)).strftime("%Y-%m-%d")
        # toDate = datetime.now(TIME_ZONE).strftime("%Y-%m-%d")
        # fromDate = "2025-04-11"
        # toDate = "2025-04-11"

        # This url string is for fetching one day intraday data at end of day.
        url = f"https://api.upstox.com/v2/historical-candle/intraday/{parseInstrument}/{timeFrame}"
        # url = f"https://api.upstox.com/v2/historical-candle/{parseInstrument}/{timeFrame}/{toDate}/{fromDate}"
        res = requests.get(
            url,
            headers={
                "accept": "application/json",
            },
            params={},
            timeout=5.0,
        )
        candleRes = res.json()

        if (
            "data" in candleRes
            and "candles" in candleRes["data"]
            and candleRes["data"]["candles"]
        ):
            candleData = pd.DataFrame(candleRes["data"]["candles"])
            candleData.columns = ["date", "open", "high", "low", "close", "vol", "oi"]
            candleData = candleData[["date", "open", "high", "low", "close"]]

            candleData["date"] = pd.to_datetime(candleData["date"]).dt.tz_convert(
                timeZone
            )

            candleData["symbol"] = tradingsymbol
            print(tradingsymbol, len(candleData))
            return candleData
        else:
            print("No data", instrument_key, candleRes)

    except Exception as e:
        print(f"Error in data fetch for {instrument_key} {res} {e}")


""" 
instrument_key = "NSE_EQ|INE040A01034"
tradingsymbol = "HDFCBANK"

res = getHistoricalData(instrument_key, tradingsymbol)

print(res)
"""

candleOfList = []
df = pd.read_csv("csv/Watchlist_equity.csv")
df["instrument_key"] = "NSE_EQ|" + df["instrument_key"].astype(str)

for index, row in df.iterrows():
    instrument_key = row["instrument_key"]
    tradingsymbol = row["tradingsymbol"]
    res = getHistoricalData(instrument_key, tradingsymbol)
    if res is not None:
        candleOfList.append(res)


df = pd.read_csv("csv/Watchlist_index.csv")
df["instrument_key"] = "NSE_INDEX|" + df["instrument_key"].astype(str)

for index, row in df.iterrows():
    instrument_key = row["instrument_key"]
    tradingsymbol = row["tradingsymbol"]
    res = getHistoricalData(instrument_key, tradingsymbol)
    if res is not None:
        candleOfList.append(res)

for symData in candleOfList:
    try:
        filename = symData.iloc[0]["symbol"]
        folder_path = "upload_csv/"
        filename = f"{filename}.csv"
        symData.to_csv(folder_path + filename, index=False)

    except Exception as e:
        print(f"Error {e}")
