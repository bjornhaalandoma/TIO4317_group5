import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import os

# Create a directory to store data
output_dir = "financial_data"
os.makedirs(output_dir, exist_ok=True)

# Define time range
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=15 * 365)  # Last 5 years

# Define tickers for Yahoo Finance
tickers = {
    "Brent_Crude": "BZ=F",       # Brent Crude Oil Futures
    "Natural_Gas": "NG=F",       # Natural Gas Futures
    "NOK_USD": "NOK=X",          # NOK/USD Exchange Rate
    "OSEBX": "^OSEAX",           # OSEBX Index
    "Var_Energi": "AKRBP.OL"       # Vår Energi Stock Price
}

# Function to fetch data and store in a single CSV per asset


def fetch_yahoo_data(ticker, name):
    try:
        # Fetch stock data
        data = yf.download(ticker, start=start_date,
                           end=end_date, progress=False, auto_adjust=False)

        if data.empty:
            print(f"Warning: No data for {name} ({ticker})")
            return None

        # Compute log returns
        data["Log_Returns"] = np.log(data["Close"] / data["Close"].shift(1))

        # Reset index to ensure 'Date' is included as a column
        data = data.reset_index()

        # Drop 'Adj Close' if it exists
        if "Adj Close" in data.columns:
            data = data.drop(columns=["Adj Close"])

        # Check if the second row is a duplicate ticker row and remove it
        # If the second row is text instead of numbers
        if isinstance(data.iloc[0, 1], str):
            data = data.iloc[1:].reset_index(drop=True)

        # Ensure column names are correctly assigned
        expected_columns = ["Date", "Close", "High",
                            "Low", "Open", "Volume", "Log_Returns"]
        if len(data.columns) == len(expected_columns):
            data.columns = expected_columns

        # Save the cleaned dataset
        file_path = os.path.join(output_dir, f"{name}.csv")
        data.to_csv(file_path, index=False)
        print(f"Saved {file_path} (Clean Data)")

        return data

    except Exception as e:
        print(f"Error fetching data for {name} ({ticker}): {e}")
        return None


# Fetch and save Yahoo Finance data
for key, ticker in tickers.items():
    fetch_yahoo_data(ticker, key)

# Fetch Interest Rates


def fetch_interest_rates():
    ir_tickers = {
        "US_10Yr_Treasury": "^TNX",  # US 10-Year Treasury Yield
    }
    for key, ticker in ir_tickers.items():
        fetch_yahoo_data(ticker, key)


fetch_interest_rates()

print("\nData collection complete. Each asset has its own CSV file in 'financial_data'. Existing files are overwritten.")

