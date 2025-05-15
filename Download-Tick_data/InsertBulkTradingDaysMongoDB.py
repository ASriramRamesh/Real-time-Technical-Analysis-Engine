# If we are working on 250 candles then we would have to fetch 2 days information for 3 min charts
# for 5 min charts we would need 3 working days
# for 15 min charts we would need 10 working days

import pandas as pd
from pymongo import MongoClient
from datetime import datetime

# I have not retained the python code to get unique trading days from 1 minute csv files.


# Function to upload trading days from the unique_dates.csv file to MongoDB
def upload_trading_days_to_timeseries(csv_file, db_name, collection_name):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Convert the 'date' column to datetime (format: YYYY-mm-dd)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    # Prepare the MongoDB documents
    documents = [{"date": row["date"]} for _, row in df.iterrows()]

    # Connect to the MongoDB server
    client = MongoClient("mongodb://localhost:27017/")  # Adjust URI if necessary
    db = client[db_name]

    # Create the time series collection if it does not exist
    if collection_name not in db.list_collection_names():
        db.create_collection(
            collection_name,
            timeseries={
                "timeField": "date",
                "granularity": "hours",  # Use 'hours' as the closest to daily granularity
            },
        )
        print(f"Time series collection '{collection_name}' created successfully.")
    else:
        print(f"Time series collection '{collection_name}' already exists.")

    # Insert the documents into the MongoDB collection
    collection = db[collection_name]
    collection.insert_many(documents)

    # Add an index on the 'date' field
    collection.create_index([("date", 1)])
    print(
        f"Data inserted and index created on 'date' for collection '{collection_name}'."
    )


# Example usage
csv_file = "unique_dates.csv"  # Path to the unique_dates.csv file
db_name = "nse_stock_data"  # Database name
collection_name = "trading_days"  # Collection name

upload_trading_days_to_timeseries(csv_file, db_name, collection_name)
