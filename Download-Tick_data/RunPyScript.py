import os
import csv


def create_stock_list_csv(input_folder, output_file):
    # Get list of all CSV files in the input folder
    csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

    # Extract stock names (file names without .csv extension)
    stock_names = [os.path.splitext(f)[0] for f in csv_files]

    # Write stock names to a new CSV file
    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["stock"])  # Header
        for stock in stock_names:
            writer.writerow([stock])

    print(f"Stock list has been saved to {output_file}")
    print(f"Total number of stocks: {len(stock_names)}")


# Usage
input_folder = "upload_csv"
output_file = "stock_list.csv"
create_stock_list_csv(input_folder, output_file)
