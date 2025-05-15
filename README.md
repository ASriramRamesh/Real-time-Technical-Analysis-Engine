# Intraday Market Data Pipeline & Real-Time Technical Analysis Engine

This repository contains two Python-based projects designed to automate the collection, storage, and real-time analysis of intraday stock market data from the National Stock Exchange (NSE) using the Upstox API.

---

## 📌 Project Overview

### Project 1: Intraday Market Data Pipeline

**Objective:**  
To fetch and store 1-minute intraday OHLC (Open, High, Low, Close) data for 80 NSE stocks, 10 sector indices, and the Nifty index during a trading day.

**Features:**
- Fetches intraday data from Upstox for all configured symbols.
- Saves raw data to CSV files.
- Pushes data into a MongoDB collection for persistence and further processing.

**Technologies Used:**  
- Python  
- Requests / Upstox API  
- CSV  
- MongoDB  

---

### Project 2: Real-Time Technical Analysis Engine

**Objective:**  
To simulate a real-time trading environment using stored intraday data and apply technical analysis indicators in a streaming fashion.

**Features:**
- Loads the latest intraday data from MongoDB.
- Converts raw 1-minute data into a nested time-interval structure:
  - Rows represent individual stocks.
  - Columns include OHLC values for specific time intervals (e.g., 9:15 AM to 3:30 PM).
- Adds technical indicators such as:
  - RSI (Relative Strength Index)
  - Supertrend
- Converts batch-processing TA libraries into simulated streaming analysis.
- Emulates live feeds for updated technical signals.

**Technologies Used:**  
- Python  
- Pandas  
- TA-Lib / Custom TA functions  
- MongoDB  

---

## 🛠️ How to Run

### Prerequisites
- Python 3.x  
- MongoDB (local or remote instance)  
- Required Python packages (see `requirements.txt`)  

### Installation
```bash
git clone https://github.com/your-username/intraday-analysis-engine.git
cd intraday-analysis-engine
pip install -r requirements.txt
