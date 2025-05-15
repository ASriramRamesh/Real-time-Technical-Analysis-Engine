import os
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import pytz

# Timezone for conversion
kolkata_tz = pytz.timezone("Asia/Kolkata")

# If we are inserting bulk dates then we need to comment the code to insert a single trading day data
collection_name = "trading_days"  # Collection name
new_trading_date = "2025-04-30"  # When uploading intraday data make sure to change this date. Format is YYYY-mm-dd


# Function to ensure an index is created only once
def ensure_datetime_index(collection):
    # Check if the index on 'datetime' already exists
    existing_indexes = collection.index_information()

    # Only create the index if it doesn't already exist
    if "datetime_1" not in existing_indexes:
        collection.create_index([("datetime", 1)])
        print("Index on 'datetime' created successfully.")
    else:
        print("Index on 'datetime' already exists.")


# Function to upload each CSV file to the corresponding MongoDB time series collection
def upload_all_csv_to_timeseries_mongodb(csv_folder, db_name):
    # Connect to the MongoDB server
    client = MongoClient("mongodb://localhost:27017/")  # Adjust URI if necessary
    db = client[db_name]

    # Loop through each CSV file in the folder
    for csv_file in os.listdir(csv_folder):
        if csv_file.endswith(".csv"):
            # Extract the stock name (filename without extension) to use for collection naming
            stock_name = csv_file.replace(".csv", "")
            collection_name = f"stock_{stock_name}"

            # Read the CSV file
            file_path = os.path.join(csv_folder, csv_file)
            df = pd.read_csv(file_path)

            # Convert the 'date' column to datetime and adjust for the Asia/Kolkata timezone
            df["date"] = pd.to_datetime(df["date"]).dt.tz_convert(kolkata_tz)

            # Prepare the MongoDB documents (no metadata or symbol column)
            documents = []
            for _, row in df.iterrows():
                documents.append(
                    {
                        "datetime": row["date"],
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                    }
                )

            # Create the time series collection if it does not exist
            if collection_name not in db.list_collection_names():
                db.create_collection(
                    collection_name,
                    timeseries={
                        "timeField": "datetime",
                        "granularity": "minutes",  # Granularity is set to minutes
                    },
                )
                print(
                    f"Time series collection '{collection_name}' created successfully."
                )
            else:
                print(f"Time series collection '{collection_name}' already exists.")

            # Insert the documents into the MongoDB collection
            collection = db[collection_name]
            collection.insert_many(documents)

            # Ensure the index is created only if it doesn't already exist
            ensure_datetime_index(collection)

            print(
                f"Data inserted and index created on 'datetime' for collection '{collection_name}'."
            )


# Example usage
csv_folder = "upload_csv"  # Folder where your CSV files are stored
db_name = "nse_stock_data"  # Database name

upload_all_csv_to_timeseries_mongodb(csv_folder, db_name)


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
