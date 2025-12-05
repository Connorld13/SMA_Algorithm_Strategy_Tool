# fetchdata.py

import yfinance as yf
import pandas as pd
from datetime import timedelta, datetime
import requests
import time
import os
from pathlib import Path

def fetch_and_prepare_data(tickers, start_date, end_date, num_rows):
    """
    Fetch and prepare stock data for the given tickers and date range.
    If start_date is None, fetches all available data up to end_date.
    If num_rows is None, no row-based limit is applied.
    """
    try:
        print(f"Fetching data for tickers: {tickers}")

        # Convert end_date to datetime and add one day for inclusivity
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
        end_date_inclusive = (end_date_dt + timedelta(days=1)).strftime("%Y-%m-%d")

        # If start_date is None, we fetch data “from inception” up to end_date_inclusive.
        # yfinance allows start=None to fetch max available history.
        # Alternatively, you can pass period="max", but we’ll do the logic below.
        yf_start = start_date if start_date else None

        data_list = []
        batch_size = 100
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i + batch_size]
            batch_data = yf.download(
                batch_tickers, 
                start=yf_start, 
                end=end_date_inclusive, 
                group_by="ticker", 
                progress=False, 
                threads=True
            )
            if batch_data.empty:
                continue

            # Handle single-ticker vs multi-ticker shape
            if len(batch_tickers) == 1:
                batch_data.columns = batch_data.columns.droplevel(0)
                batch_data = batch_data.reset_index()
                batch_data["Ticker"] = batch_tickers[0]
            else:
                batch_data = batch_data.stack(level=0).rename_axis(["Date", "Ticker"]).reset_index()

            data_list.append(batch_data)

        if not data_list:
            raise ValueError(f"No data found for tickers: {tickers}")

        data = pd.concat(data_list, ignore_index=True)

        if "Close" not in data.columns:
            raise ValueError("The 'Close' column is missing in the fetched data.")

        if not pd.api.types.is_datetime64_any_dtype(data["Date"]):
            data["Date"] = pd.to_datetime(data["Date"])

        data["Close"] = data["Close"].round(3)

        # Limit data to num_rows per ticker if num_rows is specified
        if num_rows is not None:
            limited_data = []
            for ticker in tickers:
                ticker_data = data[data["Ticker"] == ticker].copy()
                ticker_data = ticker_data.sort_values("Date").tail(num_rows)
                limited_data.append(ticker_data)
            data = pd.concat(limited_data, ignore_index=True)

        return data[["Date", "Close", "Ticker"]]
    except Exception as e:
        print(f"Error: {e}")
        raise ValueError(f"Error fetching data: {e}")

def fetch_sp500_tickers_from_api():
    """
    Fetch S&P 500 tickers and names from various free APIs.
    Tries multiple sources for reliability.
    """
    try:
        # Method 1: Try Wikipedia (most reliable free source)
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_table = tables[0]  # First table contains the S&P 500 list
            sp500_data = pd.DataFrame({
                'Symbol': sp500_table['Symbol'],
                'Name': sp500_table['Security']
            })
            print("Successfully fetched S&P 500 from Wikipedia")
            return sp500_data
        except Exception as e:
            print(f"Wikipedia method failed: {e}")
        
        # Method 2: Try using SPY ETF holdings (may not have all 500, but close)
        try:
            spy = yf.Ticker("SPY")
            # Get top holdings - this is limited but better than nothing
            # Note: This might not give all 500 companies
            print("Attempting to fetch from SPY holdings...")
            # This is a fallback, but SPY holdings API is limited
            raise Exception("SPY holdings method not fully implemented")
        except Exception as e:
            print(f"SPY method failed: {e}")
        
        # Method 3: Try financialmodelingprep API (if you have free API key)
        # This requires an API key, so we'll skip it for now
        
        raise Exception("All API methods failed")
        
    except Exception as e:
        print(f"Error fetching S&P 500 from API: {e}")
        return None

def check_ticker_compatibility(tickers_df):
    """
    Check which tickers are compatible with yfinance (have available data).
    Similar to datafetchtest.py compatibility check.
    
    Parameters:
        tickers_df: DataFrame with 'Symbol' and 'Name' columns
    
    Returns:
        DataFrame with only compatible tickers (Symbol and Name columns)
    """
    if tickers_df is None or tickers_df.empty:
        return tickers_df
    
    if 'Symbol' not in tickers_df.columns:
        return tickers_df
    
    print(f"\nChecking compatibility of {len(tickers_df)} tickers with yfinance...")
    compatible_symbols = []
    compatible_names = []
    not_compatible_count = 0
    
    tickers_list = tickers_df['Symbol'].tolist()
    names_list = tickers_df['Name'].tolist() if 'Name' in tickers_df.columns else [None] * len(tickers_list)
    
    for i, (ticker, name) in enumerate(zip(tickers_list, names_list), 1):
        try:
            # Try to fetch a small amount of data to check if ticker exists
            ticker_obj = yf.Ticker(ticker)
            # Try to get recent data (1 day should be enough to check)
            hist = ticker_obj.history(period="1d")
            
            if hist.empty:
                not_compatible_count += 1
                if i % 100 == 0:  # Print progress every 100 stocks
                    print(f"[{i}/{len(tickers_list)}] Checking... ({not_compatible_count} incompatible so far)")
            else:
                compatible_symbols.append(ticker)
                compatible_names.append(name if name is not None else ticker)
                if i % 100 == 0:  # Print progress every 100 stocks
                    print(f"[{i}/{len(tickers_list)}] Checking... ({len(compatible_symbols)} compatible so far)")
        except Exception as e:
            not_compatible_count += 1
            if i % 100 == 0:  # Print progress every 100 stocks
                print(f"[{i}/{len(tickers_list)}] Checking... ({not_compatible_count} incompatible so far)")
    
    # Create DataFrame with compatible tickers
    compatible_df = pd.DataFrame({
        'Symbol': compatible_symbols,
        'Name': compatible_names
    })
    
    print(f"\nCompatibility check complete:")
    print(f"  Compatible: {len(compatible_df)} ({len(compatible_df)/len(tickers_df)*100:.1f}%)")
    print(f"  Not compatible: {not_compatible_count} ({not_compatible_count/len(tickers_df)*100:.1f}%)")
    
    return compatible_df

def fetch_russell3000_tickers_from_api():
    """
    Fetch Russell 3000 tickers and names from iShares IWV ETF holdings CSV.
    This is the primary method as it provides the most accurate and up-to-date list.
    Uses caching to only fetch twice per month (every ~15 days).
    Filters out tickers that are not compatible with yfinance before saving to CSV.
    """
    # Cache file path
    cache_file = "russell3000_companies.csv"
    cache_metadata_file = "russell3000_cache_metadata.txt"
    
    # Check if cached file exists and is recent enough (within 15 days)
    if os.path.exists(cache_file):
        try:
            # Check last modified date of cache file
            cache_mtime = os.path.getmtime(cache_file)
            cache_date = datetime.fromtimestamp(cache_mtime)
            days_since_cache = (datetime.now() - cache_date).days
            
            # Also check metadata file for explicit fetch date
            fetch_date = None
            if os.path.exists(cache_metadata_file):
                try:
                    with open(cache_metadata_file, 'r') as f:
                        fetch_date_str = f.read().strip()
                        fetch_date = datetime.strptime(fetch_date_str, "%Y-%m-%d")
                        days_since_cache = (datetime.now() - fetch_date).days
                except:
                    pass
            
            # Only use cache if it's less than 15 days old (twice per month)
            if days_since_cache < 15:
                print(f"Using cached Russell 3000 data (from {days_since_cache} days ago)")
                cached_data = pd.read_csv(cache_file)
                if "Symbol" in cached_data.columns and "Name" in cached_data.columns:
                    return cached_data[["Symbol", "Name"]]
                else:
                    print("Cached file format invalid, will fetch new data")
            else:
                print(f"Cached data is {days_since_cache} days old, fetching new data...")
        except Exception as e:
            print(f"Error reading cache: {e}, will fetch new data")
    
    # If we get here, we need to fetch new data
    try:
        # Method 1: Fetch from iShares IWV (Russell 3000 ETF) holdings CSV
        # This is the most reliable source for Russell 3000 constituents
        try:
            print("Attempting to fetch Russell 3000 from iShares IWV ETF holdings...")
            
            # Generate date string - try recent dates (iShares data may not be available for today)
            # Start with a few days back and go further back if needed
            resp = None
            successful_date = None
            
            # Try dates from 1 day back to 60 days back (to find the most recent available data)
            for days_back in range(1, 61):  # Try up to 60 days back
                try_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
                url = f"https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund&asOfDate={try_date}"
                
                try:
                    test_resp = requests.get(url, timeout=30)
                    # Check if we got a substantial response (actual data is usually > 100KB)
                    if test_resp.status_code == 200 and len(test_resp.content) > 100000:
                        resp = test_resp
                        successful_date = try_date
                        print(f"Successfully fetched data for date: {try_date} (size: {len(resp.content)} bytes)")
                        break
                except Exception as e:
                    continue
            
            if resp is None:
                raise Exception("Could not fetch iShares data for any recent date")
            
            if resp.status_code == 200 and len(resp.content) > 100000:
                # Parse the CSV content
                from io import StringIO
                # Handle BOM and encoding issues
                try:
                    csv_content = resp.content.decode('utf-8-sig')  # utf-8-sig handles BOM
                except:
                    csv_content = resp.content.decode('utf-8', errors='ignore')
                
                # Save raw CSV for debugging first
                with open("russell3000_raw_debug.csv", "wb") as f:
                    f.write(resp.content)
                print("Saved raw CSV to russell3000_raw_debug.csv for inspection")
                
                # iShares CSV often has metadata rows at the top that need to be skipped
                # Try to find where the actual data starts by looking for header row
                lines = csv_content.split('\n')
                data_start_line = 0
                data_end_line = len(lines)
                header_found = False
                
                # Look for header row (usually contains "Ticker" or "Symbol")
                for i, line in enumerate(lines[:30]):  # Check first 30 lines
                    line_lower = line.lower()
                    if ('ticker' in line_lower or 'symbol' in line_lower) and 'name' in line_lower:
                        data_start_line = i
                        header_found = True
                        print(f"Found header row at line {i+1}")
                        break
                
                # Look for footer/legal text that marks the end of data
                # Common markers: "The content contained herein", "CAREFULLY CONSIDER", copyright symbols
                for i, line in enumerate(lines[data_start_line:], start=data_start_line):
                    line_lower = line.lower().strip()
                    if (line_lower.startswith('"the content contained herein') or 
                        line_lower.startswith('carefully consider') or
                        'blackrock, inc' in line_lower and 'rights reserved' in line_lower):
                        data_end_line = i
                        print(f"Found footer/legal text at line {i+1}, stopping data extraction")
                        break
                
                # Extract only the data section (skip header metadata and footer legal text)
                data_section = '\n'.join(lines[data_start_line:data_end_line])
                
                # Try multiple parsing strategies
                df = None
                parsing_errors = []
                
                # Strategy 1: Read with skiprows if we found the header
                if header_found:
                    try:
                        df = pd.read_csv(StringIO(data_section), encoding='utf-8', 
                                       on_bad_lines='skip', engine='python')
                        print(f"Successfully parsed CSV (lines {data_start_line+1} to {data_end_line})")
                    except Exception as e1:
                        parsing_errors.append(f"Strategy 1 (skiprows) failed: {e1}")
                
                # Strategy 2: Try reading with python engine and error handling
                if df is None or df.empty:
                    try:
                        df = pd.read_csv(StringIO(data_section), encoding='utf-8', on_bad_lines='skip', 
                                       engine='python', skipinitialspace=True)
                        print("Successfully parsed CSV with python engine")
                    except Exception as e2:
                        parsing_errors.append(f"Strategy 2 (python engine) failed: {e2}")
                
                # Strategy 3: Try with different quote handling
                if df is None or df.empty:
                    try:
                        df = pd.read_csv(StringIO(data_section), encoding='utf-8', quotechar='"', 
                                       skipinitialspace=True, on_bad_lines='skip', engine='python',
                                       sep=',')
                        print("Successfully parsed CSV with quote handling")
                    except Exception as e3:
                        parsing_errors.append(f"Strategy 3 (quote handling) failed: {e3}")
                
                # Strategy 4: Manual parsing if all else fails
                if df is None or df.empty:
                    print("All automatic parsing strategies failed, trying manual parsing...")
                    print(f"Parsing errors: {parsing_errors}")
                    # Try to manually extract data (only from data section, not footer)
                    data_rows = []
                    for i, line in enumerate(lines[data_start_line:data_end_line], 
                                          start=data_start_line):
                        if not line.strip() or line.startswith('Fund') or line.startswith('As of'):
                            continue
                        # Skip footer markers
                        line_lower = line.lower().strip()
                        if (line_lower.startswith('"the content contained herein') or 
                            line_lower.startswith('carefully consider')):
                            break
                        # Try to split by comma, handling quoted fields
                        parts = []
                        in_quotes = False
                        current_part = ""
                        for char in line:
                            if char == '"':
                                in_quotes = not in_quotes
                            elif char == ',' and not in_quotes:
                                parts.append(current_part.strip())
                                current_part = ""
                            else:
                                current_part += char
                        if current_part:
                            parts.append(current_part.strip())
                        
                        if len(parts) >= 2:
                            # Assume first column is ticker, second is name (or find them)
                            data_rows.append(parts)
                    
                    if data_rows:
                        # Try to find ticker and name columns from first row
                        header_row = data_rows[0] if data_rows else []
                        ticker_idx = None
                        name_idx = None
                        for idx, col in enumerate(header_row):
                            col_lower = str(col).lower()
                            if 'ticker' in col_lower or 'symbol' in col_lower:
                                ticker_idx = idx
                            if ('name' in col_lower or 'company' in col_lower) and ticker_idx != idx:
                                name_idx = idx
                        
                        if ticker_idx is not None and name_idx is not None:
                            # Create DataFrame from data rows
                            data_dict = {'Symbol': [], 'Name': []}
                            for row in data_rows[1:]:  # Skip header
                                if len(row) > max(ticker_idx, name_idx):
                                    data_dict['Symbol'].append(str(row[ticker_idx]).strip().upper())
                                    data_dict['Name'].append(str(row[name_idx]).strip())
                            df = pd.DataFrame(data_dict)
                            print(f"Manually parsed {len(df)} rows from CSV")
                
                if df is None or df.empty:
                    print("Failed to parse CSV with all strategies")
                    print("First 10 lines of CSV:")
                    for i, line in enumerate(lines[:10]):
                        print(f"Line {i+1}: {line[:100]}")
                    raise Exception("Could not parse iShares CSV file")
                
                # Clean column names (remove whitespace)
                df.columns = df.columns.str.strip()
                
                # iShares CSV column names can vary, try common variations
                symbol_col = None
                name_col = None
                
                # Try to find symbol column (case-insensitive)
                for col in df.columns:
                    col_lower = str(col).lower().strip()
                    if 'ticker' in col_lower or ('symbol' in col_lower and 'cusip' not in col_lower):
                        symbol_col = col
                        break
                
                # Try to find name column
                for col in df.columns:
                    col_lower = str(col).lower().strip()
                    if ('name' in col_lower or 'company' in col_lower) and 'ticker' not in col_lower and 'symbol' not in col_lower:
                        name_col = col
                        break
                
                if symbol_col and name_col:
                    # Extract and clean the data
                    russell_data = pd.DataFrame({
                        'Symbol': df[symbol_col].astype(str).str.strip().str.upper(),
                        'Name': df[name_col].astype(str).str.strip()
                    })
                    
                    # Remove any rows with empty or invalid symbols
                    russell_data = russell_data[russell_data['Symbol'].notna()]
                    russell_data = russell_data[russell_data['Symbol'] != '']
                    russell_data = russell_data[russell_data['Symbol'] != 'NAN']
                    russell_data = russell_data[~russell_data['Symbol'].str.contains('nan', case=False, na=False)]
                    russell_data = russell_data[~russell_data['Symbol'].str.contains('^[0-9]+$', regex=True)]  # Remove pure numbers
                    
                    # Remove duplicates
                    russell_data = russell_data.drop_duplicates(subset=['Symbol'], keep='first')
                    
                    if len(russell_data) > 0:
                        print(f"Successfully fetched {len(russell_data)} stocks from iShares IWV ETF holdings")
                        
                        # Check compatibility with yfinance and filter out incompatible tickers
                        russell_data = check_ticker_compatibility(russell_data)
                        
                        if len(russell_data) > 0:
                            # Save to cache file (only compatible tickers)
                            try:
                                russell_data.to_csv(cache_file, index=False)
                                # Save metadata with fetch date
                                with open(cache_metadata_file, 'w') as f:
                                    f.write(datetime.now().strftime("%Y-%m-%d"))
                                print(f"Cached {len(russell_data)} compatible Russell 3000 tickers to {cache_file}")
                            except Exception as cache_error:
                                print(f"Warning: Could not save cache: {cache_error}")
                            
                            return russell_data
                        else:
                            print("No compatible tickers found after filtering")
                    else:
                        print("No valid stocks found after parsing")
                else:
                    print(f"Could not find expected columns in CSV. Available columns: {df.columns.tolist()}")
                    print(f"Looking for symbol column, found: {symbol_col}")
                    print(f"Looking for name column, found: {name_col}")
                    # Print first few rows for debugging
                    print("First 5 rows of CSV:")
                    print(df.head())
            else:
                print(f"Failed to fetch from iShares: HTTP {resp.status_code}, content length: {len(resp.content) if resp else 0}")
                
        except Exception as e:
            print(f"iShares IWV method failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Method 2: Fallback to S&P indices combination (if iShares fails)
        try:
            print("Falling back to S&P indices combination method...")
            all_stocks = []
            
            # Get S&P 500
            try:
                sp500 = fetch_sp500_tickers_from_api()
                if sp500 is not None and not sp500.empty:
                    all_stocks.append(sp500)
                    print(f"Added {len(sp500)} stocks from S&P 500")
            except Exception as e:
                print(f"S&P 500 fetch failed: {e}")
            
            # Try to get S&P 400 (MidCap) and S&P 600 (SmallCap) from Wikipedia
            try:
                # S&P 400 MidCap
                url_mid = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
                tables_mid = pd.read_html(url_mid)
                if tables_mid and len(tables_mid) > 0:
                    sp400_table = tables_mid[0]
                    if 'Symbol' in sp400_table.columns and 'Security' in sp400_table.columns:
                        sp400_data = pd.DataFrame({
                            'Symbol': sp400_table['Symbol'],
                            'Name': sp400_table['Security']
                        })
                        all_stocks.append(sp400_data)
                        print(f"Added {len(sp400_data)} stocks from S&P 400")
            except Exception as e:
                print(f"S&P 400 fetch failed: {e}")
            
            try:
                # S&P 600 SmallCap
                url_small = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
                tables_small = pd.read_html(url_small)
                if tables_small and len(tables_small) > 0:
                    sp600_table = tables_small[0]
                    if 'Symbol' in sp600_table.columns and 'Security' in sp600_table.columns:
                        sp600_data = pd.DataFrame({
                            'Symbol': sp600_table['Symbol'],
                            'Name': sp600_table['Security']
                        })
                        all_stocks.append(sp600_data)
                        print(f"Added {len(sp600_data)} stocks from S&P 600")
            except Exception as e:
                print(f"S&P 600 fetch failed: {e}")
            
            # Combine all stocks and remove duplicates
            if all_stocks:
                combined = pd.concat(all_stocks, ignore_index=True)
                combined = combined.drop_duplicates(subset=['Symbol'], keep='first')
                print(f"Successfully fetched {len(combined)} unique stocks from S&P indices (fallback method)")
                
                if len(combined) > 0:
                    # Check compatibility with yfinance and filter out incompatible tickers
                    combined = check_ticker_compatibility(combined)
                    
                    if len(combined) > 0:
                        # Save to cache file (only compatible tickers)
                        try:
                            combined.to_csv(cache_file, index=False)
                            # Save metadata with fetch date
                            with open(cache_metadata_file, 'w') as f:
                                f.write(datetime.now().strftime("%Y-%m-%d"))
                            print(f"Cached {len(combined)} compatible Russell 3000 tickers to {cache_file}")
                        except Exception as cache_error:
                            print(f"Warning: Could not save cache: {cache_error}")
                        
                        return combined
                    else:
                        print("No compatible tickers found after filtering")
                    
        except Exception as e:
            print(f"Fallback method failed: {e}")
        
        # If all methods fail, try to return cached data even if old
        if os.path.exists(cache_file):
            try:
                print("All fetch methods failed, using cached data (even if old)...")
                cached_data = pd.read_csv(cache_file)
                if "Symbol" in cached_data.columns and "Name" in cached_data.columns:
                    return cached_data[["Symbol", "Name"]]
            except Exception as e:
                print(f"Could not load cached data: {e}")
        
        print("All methods failed to fetch Russell 3000 stocks")
        return None
        
    except Exception as e:
        print(f"Error fetching Russell 3000 from API: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to return cached data as fallback
        if os.path.exists(cache_file):
            try:
                print("Attempting to use cached data as fallback...")
                cached_data = pd.read_csv(cache_file)
                if "Symbol" in cached_data.columns and "Name" in cached_data.columns:
                    return cached_data[["Symbol", "Name"]]
            except:
                pass
        
        return None

def get_index_constituents(index_name="S&P 500"):
    """
    Get index constituents from API, with CSV fallback.
    
    Parameters:
        index_name: "S&P 500" or "Russell 3000"
    
    Returns:
        DataFrame with 'Symbol' and 'Name' columns, or None if all methods fail
    """
    if index_name == "S&P 500":
        # Try API first
        api_data = fetch_sp500_tickers_from_api()
        if api_data is not None and not api_data.empty:
            return api_data
        
        # Fallback to CSV
        try:
            csv_data = pd.read_csv("sp500_companies.csv")
            if "Symbol" in csv_data.columns and "Name" in csv_data.columns:
                print("Using S&P 500 CSV file as fallback")
                return csv_data[["Symbol", "Name"]]
        except Exception as e:
            print(f"CSV fallback failed: {e}")
        
        return None
        
    elif index_name == "Russell 3000":
        # Try API first
        api_data = fetch_russell3000_tickers_from_api()
        if api_data is not None and not api_data.empty:
            return api_data
        
        # Fallback to CSV
        try:
            csv_data = pd.read_csv("russell3000_companies.csv")
            if "Symbol" in csv_data.columns and "Name" in csv_data.columns:
                print("Using Russell 3000 CSV file as fallback")
                return csv_data[["Symbol", "Name"]]
        except Exception as e:
            print(f"CSV fallback failed: {e}")
        
        return None
    
    return None
