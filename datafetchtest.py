import yfinance as yf

# Define the ticker symbol
ticker_symbol = "NVDA"

# Create a Ticker object
ticker = yf.Ticker(ticker_symbol)

# Fetch historical market data
historical_data = ticker.history(period="5y")  # data for the last year
print("Historical Data:")
print(historical_data)
