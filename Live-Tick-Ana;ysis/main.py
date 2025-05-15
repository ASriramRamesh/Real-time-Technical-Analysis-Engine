from fastapi import FastAPI, HTTPException  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.responses import StreamingResponse  # type: ignore
from datetime import datetime, timedelta
from typing import List, Dict, Union
from pydantic import BaseModel  # type: ignore
import asyncio
import random
import json
import pandas as pd
from pymongo import MongoClient
import uvicorn  # type: ignore
import pytz

from proxyendpoint import get_chart_data_response


class ChartDataRequest(BaseModel):
    datetime: str
    sector: int
    timeframe: int
    indicator: int


class ChartDataResponse(BaseModel):
    symbol: str
    data: Dict[str, float]


app = FastAPI()

# Flag to control streaming
streaming_active = False

origins = [
    "http://localhost:3000",  # Allow React app
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sectors = ["Technology", "Healthcare", "Finance"]
stocks = {
    "Technology": ["AAPL", "MSFT", "GOOGL"],
    "Healthcare": ["JNJ", "UNH", "PFE"],
    "Finance": ["JPM", "BAC", "WFC"],
}


@app.post("/chart-data")
async def get_chart_data(
    request: ChartDataRequest,
) -> List[Dict[str, Union[str, float]]]:
    try:

        utc_datetime = datetime.fromisoformat(request.datetime.replace("Z", "+00:00"))
        ist_timezone = pytz.timezone("Asia/Kolkata")

        # Convert to IST
        input_datetime = utc_datetime.astimezone(ist_timezone)

        # If your pandas DataFrame index is timezone-naive, remove timezone info
        input_datetime_naive = input_datetime.replace(tzinfo=None)
        start_datetime_naive = pd.to_datetime(input_datetime_naive)
        input_time_frame = request.timeframe
        input_indicator = request.indicator
        input_sector = request.sector

        result_list: List[Dict[str, Union[str, float]]] = get_chart_data_response(
            start_datetime_naive, input_time_frame, input_indicator, input_sector
        )

        return result_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def connect_to_mongodb(host="localhost", port=27017):
    client = MongoClient(f"mongodb://{host}:{port}/")
    return client


# This returns the days for which we have up loaded intraday data.
def get_trading_days_data(db, limit=300):
    collection_name = "trading_days"
    trading_days_col = db[collection_name]
    return pd.DataFrame(list(trading_days_col.find().sort("date", -1).limit(limit)))


@app.get("/trading-dates")
def get_trading_dates() -> List[str]:
    client = connect_to_mongodb()
    db = client["nse_stock_data"]

    df_trading_days = get_trading_days_data(db)
    client.close()

    return (df_trading_days["date"].dt.date - timedelta(days=1)).astype(str).tolist()


if __name__ == "__main__":
    uvicorn.run(app="app.main:app", host="localhost", port=8000, reload=True)
