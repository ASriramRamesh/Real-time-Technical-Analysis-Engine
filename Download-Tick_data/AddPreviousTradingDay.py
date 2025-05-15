import os
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import pytz

# Timezone for conversion
kolkata_tz = pytz.timezone("Asia/Kolkata")

# If we are inserting bulk dates then we need to comment the code to insert a single trading day data
collection_name = "trading_days"  # Collection name
new_trading_date = "2025-02-01"  # When uploading intraday data make sure to change this date. Format is YYYY-mm-dd


# Function to insert a new trading date into the trading_days collection
def insert_trading_date(db_name, collection_name, trading_date_str):
    # Convert the trading date string (in YYYY-mm-dd format) to a datetime object
    trading_date = datetime.strptime(trading_date_str, "%Y-%m-%d")

    # Connect to the MongoDB server
    client = MongoClient("mongodb://localhost:27017/")  # Adjust URI if necessary
    db = client[db_name]
    collection = db[collection_name]

    # Create the document with the new trading date
    document = {"date": trading_date}

    # Insert the new document into the collection
    collection.insert_one(document)
    print(
        f"New trading date {trading_date_str} inserted successfully into collection '{collection_name}'."
    )


# Example usage
db_name = "nse_stock_data"  # Database name

insert_trading_date(db_name, collection_name, new_trading_date)
