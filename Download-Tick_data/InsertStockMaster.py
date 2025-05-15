import os
import pandas as pd
from pymongo import MongoClient


# Function to upload trading days from the unique_dates.csv file to MongoDB
def upload_stock_master_data(csv_folder, csv_file, db_name, collection_name):
    # Read the CSV file
    file_path = os.path.join(csv_folder, csv_file)
    df = pd.read_csv(file_path)

    client = MongoClient("mongodb://localhost:27017/")  # Adjust URI if necessary
    db = client[db_name]

    # Prepare the MongoDB documents (no metadata or symbol column)
    documents = []
    for _, row in df.iterrows():
        documents.append(
            {
                "stock_id": row["stock_id"],
                "stock_symbol": row["stock_symbol"],
                "instrument_key": row["instrument_key"],
                "sector_id": row["sector_id"],
            }
        )

    # Insert the documents into the MongoDB collection
    collection = db[collection_name]
    collection.insert_many(documents)

    print(f"Collection '{collection_name}' has been created.")


csv_folder = "csv"  # Path of folder where we have the csv file
csv_file = "stock_master.csv"  # csv file name for uploading stock master collection
db_name = "nse_stock_data"  # Database name
collection_name = "stock_master"  # Collection name

upload_stock_master_data(csv_folder, csv_file, db_name, collection_name)
