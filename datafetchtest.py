import yfinance as yf
import fetchdata
from datetime import datetime, timedelta

# Get Russell 3000 tickers
print("Fetching Russell 3000 ticker list...")
russell_data = fetchdata.get_index_constituents("Russell 3000")

if russell_data is None or russell_data.empty:
    print("Error: Could not fetch Russell 3000 tickers")
    exit(1)

tickers = russell_data['Symbol'].tolist()
print(f"Found {len(tickers)} tickers to check\n")
print("Checking compatibility with yfinance...\n")

# Track results
compatible = []
not_compatible = []
errors = {}

# Check each ticker
for i, ticker in enumerate(tickers, 1):
    try:
        # Try to fetch a small amount of data to check if ticker exists
        ticker_obj = yf.Ticker(ticker)
        # Try to get info or recent data (1 day should be enough to check)
        hist = ticker_obj.history(period="1d")
        
        if hist.empty:
            not_compatible.append(ticker)
            errors[ticker] = "No data available"
            print(f"[{i}/{len(tickers)}] ❌ {ticker}: No data")
        else:
            compatible.append(ticker)
            if i % 50 == 0:  # Print progress every 50 stocks
                print(f"[{i}/{len(tickers)}] ✓ {ticker}: Compatible")
    except Exception as e:
        not_compatible.append(ticker)
        error_msg = str(e)[:100]  # Truncate long error messages
        errors[ticker] = error_msg
        print(f"[{i}/{len(tickers)}] ❌ {ticker}: Error - {error_msg}")

# Print summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total tickers checked: {len(tickers)}")
print(f"Compatible (have data): {len(compatible)} ({len(compatible)/len(tickers)*100:.1f}%)")
print(f"Not compatible (no data/error): {len(not_compatible)} ({len(not_compatible)/len(tickers)*100:.1f}%)")

# Print not compatible list
if not_compatible:
    print("\n" + "="*60)
    print("TICKERS NOT COMPATIBLE WITH YFINANCE:")
    print("="*60)
    for ticker in not_compatible:
        error = errors.get(ticker, "Unknown error")
        print(f"  {ticker}: {error}")

print("\n" + "="*60)
print("Check complete!")
