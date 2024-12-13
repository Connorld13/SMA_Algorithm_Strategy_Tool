# fetchdata.py

import yfinance as yf
import pandas as pd
from datetime import timedelta, datetime

def fetch_and_prepare_data(tickers, start_date, end_date, num_rows):
    """
    Fetch and prepare stock data for the given tickers and date range.

    Parameters:
        tickers (list): List of stock tickers.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        num_rows (int): Number of rows to limit per ticker.

    Returns:
        pd.DataFrame: A DataFrame with historical stock data.
    """
    try:
        print(f"Fetching data for tickers: {tickers}")

        # Convert end_date to datetime and add one day for inclusivity
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
        end_date_inclusive = (end_date_dt + timedelta(days=1)).strftime("%Y-%m-%d")

        # Fetch data in batches to handle large number of tickers
        data_list = []
        batch_size = 100  # Adjust batch size as needed
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i + batch_size]
            batch_data = yf.download(batch_tickers, start=start_date, end=end_date_inclusive, group_by="ticker", progress=False, threads=True)
            if batch_data.empty:
                continue
            if len(batch_tickers) == 1:
                # Flatten MultiIndex columns for single ticker
                batch_data.columns = batch_data.columns.droplevel(0)
                batch_data = batch_data.reset_index()
                batch_data["Ticker"] = batch_tickers[0]
            else:
                # Flatten MultiIndex for multiple tickers
                batch_data = batch_data.stack(level=0).rename_axis(["Date", "Ticker"]).reset_index()
            data_list.append(batch_data)

        if not data_list:
            raise ValueError(f"No data found for tickers: {tickers}")

        data = pd.concat(data_list, ignore_index=True)

        # Ensure 'Close' column is present
        if "Close" not in data.columns:
            raise ValueError("The 'Close' column is missing in the fetched data.")

        # Convert 'Date' column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(data['Date']):
            data['Date'] = pd.to_datetime(data['Date'])

        # **Temporary Change: Round 'Close' prices to thousandths place**
        data['Close'] = data['Close'].round(3)

        # Limit data to num_rows per ticker
        limited_data = []
        for ticker in tickers:
            ticker_data = data[data['Ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('Date').tail(num_rows)
            limited_data.append(ticker_data)

        data = pd.concat(limited_data, ignore_index=True)

        return data[['Date', 'Close', 'Ticker']]
    except Exception as e:
        print(f"Error: {e}")
        raise ValueError(f"Error fetching data: {e}")
