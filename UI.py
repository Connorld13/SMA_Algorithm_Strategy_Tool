# UI.py

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import threading
import os
import subprocess
import sys
import json
import urllib.request

# Try to import version checker (Python 3.8+)
try:
    from importlib.metadata import version as get_package_version
except ImportError:
    # Fallback for older Python versions
    try:
        from importlib_metadata import version as get_package_version
    except ImportError:
        # Last resort: use pkg_resources
        try:
            import pkg_resources
            def get_package_version(package_name):
                return pkg_resources.get_distribution(package_name).version
        except ImportError:
            def get_package_version(package_name):
                return None

from fetchdata import fetch_and_prepare_data
import algorithm
import scoring
import scoring

# ---- NEW IMPORT ----
import visual  # <-- (1) Import the new visual.py
import cache_manager  # Import cache manager for viewing cached results

# Global variable to store algorithm results
algorithm_results = {}

# Global scoring configuration
scoring_config = scoring.get_default_scoring_config()

# Declare global widgets to be accessed by set_status and update_progress
root = None
status_label = None
progress_bar = None

# Add new global variable for cancel flags
cancel_requested = False

# Persistent set to track selected stocks across filter operations
persistent_selected_stocks = set()

def check_library_updates():
    """Check for yfinance updates before startup using PyPI API."""
    print("Checking for yfinance updates...", end="", flush=True)
    try:
        # Get installed version
        try:
            installed_version = get_package_version('yfinance')
            if installed_version is None:
                print(" (could not get installed version)")
                return
        except Exception:
            print(" (could not get installed version)")
            return
        
        # Query PyPI JSON API (fast and reliable)
        url = "https://pypi.org/pypi/yfinance/json"
        with urllib.request.urlopen(url, timeout=5) as response:
            pypi_data = json.loads(response.read())
            latest_version = pypi_data['info']['version']
        
        # Compare versions
        if installed_version != latest_version:
            print(f"\n{'='*60}")
            print("Library Update Check:")
            print("="*60)
            print(f"yfinance update available: {installed_version} -> {latest_version}")
            print("Updating yfinance...")
            print("="*60)
            
            # Automatically update yfinance
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', '--upgrade', 'yfinance'],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    print("[SUCCESS] yfinance updated successfully!")
                    # Reload the module if it was already imported
                    try:
                        import yfinance
                        import importlib
                        importlib.reload(yfinance)
                    except:
                        pass
                else:
                    print(f"[FAILED] Update failed: {result.stderr[:200]}")
                    print("You can manually update with: pip install --upgrade yfinance")
            except subprocess.TimeoutExpired:
                print("[FAILED] Update timed out. You can manually update with: pip install --upgrade yfinance")
            except Exception as e:
                print(f"[FAILED] Update error: {str(e)[:100]}")
                print("You can manually update with: pip install --upgrade yfinance")
            print("="*60 + "\n")
        else:
            print(" (up to date)")
    except urllib.error.URLError:
        print(" (could not connect to PyPI)")
    except Exception as e:
        print(f" (error: {str(e)[:50]})")
        # Continue execution even if update check fails

def fetch_sp500_tickers_from_csv():
    """Fetch S&P 500 tickers and names. Tries API first, then CSV fallback."""
    from fetchdata import get_index_constituents
    
    # Try to get from API
    api_data = get_index_constituents("S&P 500")
    if api_data is not None and not api_data.empty:
        return api_data
    
    # Fallback to CSV
    try:
        sp500_data = pd.read_csv("sp500_companies.csv")
        return sp500_data
    except Exception as e:
        messagebox.showwarning("Warning", f"Failed to fetch S&P 500 tickers from CSV: {e}\nTrying API...")
        # Try API one more time
        api_data = get_index_constituents("S&P 500")
        if api_data is not None and not api_data.empty:
            return api_data
        messagebox.showerror("Error", "Failed to fetch S&P 500 tickers from both API and CSV.")
        return pd.DataFrame(columns=["Symbol", "Name"])

def fetch_russell3000_tickers():
    """Fetch Russell 3000 tickers and names dynamically. Tries API first, then CSV fallback."""
    from fetchdata import get_index_constituents
    
    # Try to get from API (dynamic fetch)
    api_data = get_index_constituents("Russell 3000")
    if api_data is not None and not api_data.empty:
        print(f"Successfully fetched {len(api_data)} stocks from API for Russell 3000")
        return api_data
    
    # Fallback to CSV only if API fails
    try:
        if os.path.exists("russell3000_companies.csv"):
            russell_data = pd.read_csv("russell3000_companies.csv")
            if "Symbol" in russell_data.columns and "Name" in russell_data.columns:
                print(f"Using CSV fallback with {len(russell_data)} stocks")
                return russell_data[["Symbol", "Name"]]
    except Exception as e:
        print(f"CSV read failed: {e}")
    
    # If both API and CSV fail, show a warning but continue with empty list
    # The user can still add custom stocks
    messagebox.showwarning(
        "Russell 3000 Fetch Warning",
        "Could not fetch Russell 3000 list dynamically.\n\n"
        "The system tried to fetch from free APIs but was unsuccessful.\n"
        "You can:\n"
        "- Add custom stocks manually\n"
        "- Create a 'russell3000_companies.csv' file with 'Symbol' and 'Name' columns\n"
        "- Switch to S&P 500 which is available dynamically"
    )
    return pd.DataFrame(columns=["Symbol", "Name"])

def on_index_change():
    """Update stock list when index selection changes."""
    global stock_data, persistent_selected_stocks
    index_selection = index_var.get()
    
    if index_selection == "S&P 500":
        stock_data = fetch_sp500_tickers_from_csv()
    elif index_selection == "Russell 3000":
        stock_data = fetch_russell3000_tickers()
    else:
        stock_data = fetch_sp500_tickers_from_csv()  # Default
    
    # Refresh the table
    search_entry.delete(0, tk.END)
    
    # Update mode if in "Entire" mode to match selected index
    mode = mode_var.get()
    if mode == "Entire S&P 500" and index_selection == "Russell 3000":
        mode_var.set("Entire Russell 3000")
        on_mode_change()
    elif mode == "Entire Russell 3000" and index_selection == "S&P 500":
        mode_var.set("Entire S&P 500")
        on_mode_change()
    else:
        # Clear persistent selections when switching indices (unless in Entire mode)
        persistent_selected_stocks.clear()
        filter_table("")

def on_mode_change():
    """Update UI based on the selected mode."""
    global visuals_checkbox, stock_data, persistent_selected_stocks  # Add global declaration
    mode = mode_var.get()
    search_entry.delete(0, tk.END)
    filter_table("")

    # Always show the stock selection table and search frame
    # These are now managed by pack, so no need to show/hide them
    
    if mode == "Entire S&P 500" or mode == "Entire Russell 3000":
        # Automatically check all stocks
        for row in tree.get_children():
            tree.set(row, "Select", "✓")
        # Update persistent set to include all visible stocks
        persistent_selected_stocks = {tree.item(r)["values"][0] for r in tree.get_children()}
    else:
        # Clear all selections when switching modes (but keep persistent set for filter restoration)
        for row in tree.get_children():
            tree.set(row, "Select", "")
        # Only clear persistent set if switching to Single Stock mode (which requires only one selection)
        if mode == "Single Stock":
            persistent_selected_stocks.clear()
            
    # Update the visuals checkbox state
    if mode == "Single Stock":
        visuals_checkbox.configure(state="normal")
    else:
        show_visuals_var.set(False)
        visuals_checkbox.configure(state="disabled")

def toggle_checkbox(event):
    """Toggle the checkbox state in the Treeview."""
    global persistent_selected_stocks
    mode = mode_var.get()
    region = tree.identify_region(event.x, event.y)
    if region == "cell":
        row_id = tree.identify_row(event.y)
        col_id = tree.identify_column(event.x)
        if col_id == "#3":  # Checkbox column
            if mode == "Entire S&P 500" or mode == "Entire Russell 3000":
                # Prevent unchecking in Entire mode
                index_name = "S&P 500" if mode == "Entire S&P 500" else "Russell 3000"
                messagebox.showinfo("Info", f"All stocks are automatically selected in Entire {index_name} mode.")
                return
            current_value = tree.set(row_id, "Select")
            stock_symbol = tree.item(row_id)["values"][0]
            selected_stocks = [
                tree.item(r)["values"][0]
                for r in tree.get_children()
                if tree.set(r, "Select") == "✓"
            ]

            if mode == "Single Stock" and len(selected_stocks) >= 1 and current_value == "":
                # Allow only one checkbox for Single Stock
                for r in tree.get_children():
                    tree.set(r, "Select", "")
                tree.set(row_id, "Select", "✓")
                # Update persistent set - clear all and add only this one
                persistent_selected_stocks.clear()
                persistent_selected_stocks.add(stock_symbol)
            elif mode == "10 Stocks":
                if len(selected_stocks) >= 10 and current_value == "":
                    messagebox.showwarning("Limit Reached", "You can only select up to 10 stocks.")
                else:
                    new_value = "✓" if current_value == "" else ""
                    tree.set(row_id, "Select", new_value)
                    # Update persistent set
                    if new_value == "✓":
                        persistent_selected_stocks.add(stock_symbol)
                    else:
                        persistent_selected_stocks.discard(stock_symbol)
            else:
                # For "Multi Select" and other modes without limits
                new_value = "✓" if current_value == "" else ""
                tree.set(row_id, "Select", new_value)
                # Update persistent set
                if new_value == "✓":
                    persistent_selected_stocks.add(stock_symbol)
                else:
                    persistent_selected_stocks.discard(stock_symbol)

def validate_stock_selection():
    """Validate stock selection based on the mode."""
    mode = mode_var.get()
    selected_stocks = [
        tree.item(r)["values"][0]
        for r in tree.get_children()
        if tree.set(r, "Select") == "✓"
    ]
    
    if mode == "Single Stock" and len(selected_stocks) != 1:
        messagebox.showerror("Error", "Please select exactly one stock for Single Stock mode.")
        return None
    if mode == "10 Stocks" and len(selected_stocks) != 10:
        messagebox.showerror("Error", "Please select exactly 10 stocks for 10 Stocks mode.")
        return None
    if mode == "Multi Select":
        if len(selected_stocks) < 1:
            messagebox.showerror("Error", "Please select at least one stock for Multi Select mode.")
            return None
    if mode == "Entire S&P 500" or mode == "Entire Russell 3000":
        # Ensure all stocks are selected
        all_stocks = [tree.item(r)["values"][0] for r in tree.get_children()]
        if set(selected_stocks) != set(all_stocks):
            index_name = "S&P 500" if mode == "Entire S&P 500" else "Russell 3000"
            messagebox.showerror("Error", f"All stocks must be selected for Entire {index_name} mode.")
            return None
    return selected_stocks

def filter_table(query):
    """Filter the table based on the search query."""
    global persistent_selected_stocks
    
    # Clear the main tree
    for row in tree.get_children():
        tree.delete(row)
    
    # Get all custom stocks from custom_tree if it exists and has items
    custom_stocks = {}
    if custom_tree and custom_tree.winfo_exists():
        for row in custom_tree.get_children():
            try:
                values = custom_tree.item(row)["values"]
                if values and len(values) >= 2:
                    custom_stocks[values[0]] = values[1]
            except Exception:
                continue
    
    # Add custom stocks first
    for symbol, name in custom_stocks.items():
        if query.lower() in symbol.lower() or query.lower() in name.lower():
            # Restore selection state from persistent set
            select_value = "✓" if symbol in persistent_selected_stocks else ""
            tree.insert("", "end", values=(symbol, name, select_value), tags=("row",))
    
    # Then add index stocks (S&P 500 or Russell 3000)
    if stock_data is not None and not stock_data.empty:
        for _, row_data in stock_data.iterrows():
            if (query.lower() in str(row_data["Symbol"]).lower() or 
                query.lower() in str(row_data["Name"]).lower()):
                # Skip if this stock is already added as custom
                if row_data["Symbol"] not in custom_stocks:
                    # Restore selection state from persistent set
                    select_value = "✓" if row_data["Symbol"] in persistent_selected_stocks else ""
                    tree.insert("", "end", values=(row_data["Symbol"], row_data["Name"], select_value), tags=("row",))

def set_status(text):
    """Update the status label in a thread-safe manner."""
    if root and status_label:
        root.after(0, lambda: status_label.config(text=text))

def update_progress(value):
    """Update the progress bar."""
    if progress_bar:
        progress_bar["value"] = value
        root.update_idletasks()

def request_cancel():
    """Request cancellation of the current operation."""
    global cancel_requested
    cancel_requested = True
    set_status("Cancelling...")
    # Immediately update UI to show cancellation
    root.after(0, lambda: progress_bar.configure(value=0))
    root.after(0, lambda: log_text.insert(tk.END, "\nOperation cancelled by user.\n"))
    root.after(0, lambda: log_text.see(tk.END))

def on_run_now():
    """Fetch data and execute the strategy in a separate thread with progress tracking."""
    global cancel_requested, data  # Add data to globals
    cancel_requested = False
    data = None  # Initialize data as None
    
    # Disable the Run button and enable Cancel button
    run_button.configure(state="disabled")
    cancel_button.configure(state="normal")
    
    def run_algorithm_thread():
        global algorithm_results, cancel_requested, data  # Add data to globals
        try:
            set_status("Starting algorithm...")
            update_progress(0)

            # Check for cancellation more frequently
            if cancel_requested:
                set_status("Operation cancelled.")
                return

            mode = mode_var.get()

            # Get end date input
            end_date_str = end_date_entry.get()

            # Validate end date
            try:
                end_date_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
                yesterday = datetime.now() - timedelta(days=1)
                if end_date_dt > yesterday:
                    messagebox.showerror("Error", "End date cannot be in the future.")
                    update_progress(0)
                    set_status("Ready")
                    return
            except ValueError:
                messagebox.showerror("Error", "Please enter the end date in YYYY-MM-DD format.")
                update_progress(0)
                set_status("Ready")
                return

            # Get selected time frame (using year and month inputs)
            timeframe_all_available = timeframe_all_available_var.get()
            timeframe_years = timeframe_years_var.get()
            timeframe_months = timeframe_months_var.get()

            # Decide if we are compounding or not
            is_compounding = compounding_var.get()

            # Get optimization objective
            optimization_objective = optimization_mapping.get(optimization_objective_var.get(), "taxed_return")
            
            # Get walk-forward settings
            enable_walk_forward = enable_walk_forward_var.get()
            walk_forward_config = None
            if enable_walk_forward:
                backtest_years = backtest_years_var.get()
                backtest_months = backtest_months_var.get()
                walk_forward_years = walk_forward_years_var.get()
                walk_forward_months = walk_forward_months_var.get()
                
                walk_forward_config = {
                    "backtest_period_years": backtest_years,
                    "backtest_period_months": backtest_months,
                    "walk_forward_period_years": walk_forward_years,
                    "walk_forward_period_months": walk_forward_months,
                    "rebalance_years": 0,
                    "rebalance_months": 0,
                    "rebalance_none": True
                }

            # Calculate start_date or None + num_rows
            start_date_str = None
            num_rows = None

            if timeframe_all_available:
                start_date_str = None
                num_rows = None
            else:
                # Calculate start date from years and months
                if timeframe_years == 0 and timeframe_months == 0:
                    # If both are 0, default to 5 years
                    start_date_dt = end_date_dt - relativedelta(years=5)
                    num_rows = 252 * 5
                else:
                    start_date_dt = end_date_dt - relativedelta(years=timeframe_years, months=timeframe_months)
                    # Estimate num_rows: ~252 trading days per year, ~21 per month
                    num_rows = int(252 * timeframe_years + 21 * timeframe_months)

                start_date_str = start_date_dt.strftime("%Y-%m-%d")

            # Determine selected stocks
            if mode == "Entire S&P 500":
                if stock_data is not None and not stock_data.empty:
                    selected_stocks = stock_data['Symbol'].tolist()
                    # Ensure all checkboxes are checked
                    for row in tree.get_children():
                        tree.set(row, "Select", "✓")
                else:
                    messagebox.showerror("Error", "No S&P 500 stock data available.")
                    update_progress(0)
                    set_status("Ready")
                    return
            elif mode == "Entire Russell 3000":
                if stock_data is not None and not stock_data.empty:
                    selected_stocks = stock_data['Symbol'].tolist()
                    # Ensure all checkboxes are checked
                    for row in tree.get_children():
                        tree.set(row, "Select", "✓")
                else:
                    messagebox.showerror("Error", "No Russell 3000 stock data available. Please create russell3000_companies.csv file.")
                    update_progress(0)
                    set_status("Ready")
                    return
            else:
                selected_stocks = validate_stock_selection()
                if selected_stocks is None:
                    update_progress(0)
                    set_status("Ready")
                    return

            total_selected = len(selected_stocks)

            # Check for cancellation after each major step
            if cancel_requested:
                set_status("Operation cancelled.")
                return

            # Fetch data
            set_status("Fetching data...")
            # For walk-forward, we need more historical data
            if enable_walk_forward and walk_forward_config:
                # Calculate how far back we need to go
                backtest_period = relativedelta(
                    years=walk_forward_config['backtest_period_years'],
                    months=walk_forward_config['backtest_period_months']
                )
                walk_forward_period = relativedelta(
                    years=walk_forward_config['walk_forward_period_years'],
                    months=walk_forward_config['walk_forward_period_months']
                )
                # Estimate segments needed
                estimated_segments = 5  # Conservative estimate
                total_period = backtest_period + (walk_forward_period * estimated_segments)
                # Fetch from further back
                wf_start_date = end_date_dt - total_period
                wf_start_date_str = wf_start_date.strftime("%Y-%m-%d")
                data = fetch_and_prepare_data(selected_stocks, wf_start_date_str, end_date_str, None)
            else:
                data = fetch_and_prepare_data(selected_stocks, start_date_str, end_date_str, num_rows)
            update_progress(20)

            if cancel_requested:
                set_status("Operation cancelled.")
                return

            # Run algorithm with more frequent cancellation checks
            set_status("Running algorithm...")
            total_algorithm_progress = 80
            per_stock_progress = total_algorithm_progress / total_selected if total_selected > 0 else total_algorithm_progress

            # Determine if this is a batch run (multiple stocks)
            is_batch_run = total_selected > 1
            
            # Create batch directory if batch run
            batch_dir = None
            batch_datetime = None
            if is_batch_run:
                import cache_manager
                batch_datetime = datetime.now().isoformat()
                batch_dir = cache_manager.get_batch_cache_dir(batch_datetime)
                print(f"Batch run detected: {total_selected} stocks. Saving to batch directory: {batch_dir.name}")
            
            algorithm_results = {}
            for idx, ticker in enumerate(selected_stocks, start=1):
                if cancel_requested:
                    set_status("Operation cancelled.")
                    return

                def progress_callback(algorithm_progress, current_stock_index=idx):
                    if not cancel_requested:
                        overall_progress = 20 + (current_stock_index - 1) * per_stock_progress + (algorithm_progress / 100) * per_stock_progress
                        overall_progress = min(overall_progress, 100)
                        update_progress(overall_progress)

                set_status(f"Algorithm: {ticker} ({idx}/{total_selected})")
                ticker_data = data[data['Ticker'] == ticker].copy()
                if ticker_data.empty:
                    algorithm_results[ticker] = {"Error": "No data fetched for this ticker."}
                    progress_callback(100)
                else:
                    if enable_walk_forward and walk_forward_config:
                        # For batch runs, use batch walk forward (best scored combo only)
                        # For single stock, use regular walk forward
                        import walk_forward
                        if is_batch_run:
                            # First run regular algorithm to get best combo
                            training_result = algorithm.run_algorithm(
                                ticker_data,
                                start_amount=10000,
                                progress_callback=lambda p: progress_callback(p * 0.5),  # Use first half of progress
                                compounding=is_compounding,
                                optimization_objective=optimization_objective,
                                start_date=start_date_str,
                                end_date=end_date_str,
                                use_cache=True
                            )
                            
                            if "Error" not in training_result:
                                # Then run batch walk forward on best combo
                                result = walk_forward.run_batch_walk_forward_analysis(
                                    ticker_data,
                                    start_amount=10000,
                                    progress_callback=lambda p: progress_callback(50 + p * 0.5),  # Use second half
                                    compounding=is_compounding,
                                    optimization_objective=optimization_objective,
                                    end_date=end_date_str,
                                    backtest_period_years=walk_forward_config['backtest_period_years'],
                                    backtest_period_months=walk_forward_config['backtest_period_months'],
                                    walk_forward_period_years=walk_forward_config['walk_forward_period_years'],
                                    walk_forward_period_months=walk_forward_config['walk_forward_period_months'],
                                    scoring_config=scoring_config,
                                    training_result=training_result
                                )
                            else:
                                result = training_result
                        else:
                            # Single stock: use regular walk forward
                            result = walk_forward.run_walk_forward_analysis(
                                ticker_data,
                                start_amount=10000,
                                progress_callback=progress_callback,
                                compounding=is_compounding,
                                optimization_objective=optimization_objective,
                                end_date=end_date_str,
                                backtest_period_years=walk_forward_config['backtest_period_years'],
                                backtest_period_months=walk_forward_config['backtest_period_months'],
                                walk_forward_period_years=walk_forward_config['walk_forward_period_years'],
                                walk_forward_period_months=walk_forward_config['walk_forward_period_months'],
                                rebalance_years=walk_forward_config['rebalance_years'],
                                rebalance_months=walk_forward_config['rebalance_months'],
                                rebalance_none=walk_forward_config['rebalance_none'],
                                scoring_config=scoring_config
                            )
                        # Save walk-forward results to cache
                        if result and "Error" not in result and end_date_str:
                            import cache_manager
                            from pathlib import Path
                            import pickle
                            # Get actual start date from data for cache naming (prevents "FullRange" files)
                            walk_forward_start_date = start_date_str
                            if walk_forward_start_date is None and ticker_data is not None and not ticker_data.empty and 'Date' in ticker_data.columns:
                                actual_start = ticker_data['Date'].min()
                                if hasattr(actual_start, 'strftime'):
                                    walk_forward_start_date = actual_start.strftime("%Y-%m-%d")
                                elif isinstance(actual_start, str):
                                    walk_forward_start_date = actual_start
                            
                            # Save walk-forward cache with all walk-forward data included
                            cache_manager.save_backtest_cache(
                                ticker, walk_forward_start_date, end_date_str, is_compounding, optimization_objective,
                                10000, result.get('all_combinations', []), 
                                result.get('best_combination_idx', 0), 
                                result.get('noalgoreturn', 0),
                                besttrades=[],  # Walk-forward doesn't have individual trades
                                walk_forward_mode=True,
                                segments=result.get('segments', 0),
                                training_score=result.get('training_score', 0.0),
                                walk_forward_score=result.get('walk_forward_score', 0.0),
                                combined_score=result.get('combined_score', 0.0),
                                training_metrics=result.get('training_metrics', {}),
                                walk_forward_metrics=result.get('walk_forward_metrics', {}),
                                walk_forward_segment_trades=result.get('walk_forward_segment_trades', []),  # Store trades per segment
                                batch_dir=batch_dir  # Save to batch directory if batch run
                            )
                    else:
                        # Run regular algorithm
                        result = algorithm.run_algorithm(
                            ticker_data,
                            start_amount=10000,
                            progress_callback=progress_callback,
                            compounding=is_compounding,
                            optimization_objective=optimization_objective,
                            start_date=start_date_str,
                            end_date=end_date_str,
                            use_cache=True
                        )
                    
                    algorithm_results[ticker] = result
                    
                    # Save regular algorithm results to batch directory if batch run
                    if is_batch_run and result and "Error" not in result and end_date_str:
                        import cache_manager
                        cache_manager.save_backtest_cache(
                            ticker, start_date_str, end_date_str, is_compounding, optimization_objective,
                            10000, result.get('all_combinations', []), 
                            result.get('best_combination_idx', 0), 
                            result.get('noalgoreturn', 0),
                            besttrades=result.get('besttrades', []),
                            walk_forward_mode=False,
                            batch_dir=batch_dir
                        )
                    
                    # Debug: Print walk-forward info if enabled
                    if enable_walk_forward and walk_forward_config:
                        print(f"Walk-forward result for {ticker}:")
                        print(f"  Walk-forward mode: {result.get('walk_forward_mode', False)}")
                        print(f"  Segments: {result.get('segments', 0)}")
                        print(f"  Training score: {result.get('training_score', 0)}")
                        print(f"  Walk-forward score: {result.get('walk_forward_score', 0)}")
                        print(f"  Combined score: {result.get('combined_score', 0)}")
                    
                    progress_callback(100)

                # Check for cancellation after each stock
                if cancel_requested:
                    set_status("Operation cancelled.")
                    return

            if not cancel_requested:
                update_progress(100)
                set_status("Finalizing...")
                display_results()
                
                if not cancel_requested:
                    messagebox.showinfo(
                        "Success",
                        "Algorithm execution completed. You can view the trade tables using the 'Look at Trade Table' button."
                    )
                    set_status("Completed.")
            
        except Exception as e:
            if not cancel_requested:
                messagebox.showerror("Error", str(e))
            set_status("Ready")
        finally:
            # Re-enable the Run button and disable Cancel button
            root.after(0, lambda: run_button.configure(state="normal"))
            root.after(0, lambda: cancel_button.configure(state="disabled"))
            if cancel_requested:
                update_progress(0)
                set_status("Ready")
            else:
                update_progress(100)

    thread = threading.Thread(target=run_algorithm_thread)
    thread.start()

def display_results():
    """Display results in the log text area."""
    global data, scoring_config  # Add global declaration
    if cancel_requested:
        return
        
    log_text.delete(1.0, tk.END)
    for ticker, result in algorithm_results.items():
        log_text.insert(tk.END, f"Results for {ticker}:\n")
        if "Error" in result:
            log_text.insert(tk.END, f"  {result['Error']}\n\n")
            continue
        
        # Calculate and display score
        try:
            if result.get("walk_forward_mode", False):
                # Walk-forward mode: show all three scores
                training_score = result.get("training_score", 0.0)
                walk_forward_score = result.get("walk_forward_score", 0.0)
                combined_score = result.get("combined_score", 0.0)
                segments = result.get("segments", 0)
                log_text.insert(tk.END, f"  ⭐ Walk-Forward Analysis ({segments} segments):\n")
                log_text.insert(tk.END, f"    Training Score: {training_score:.2f}/10.0\n")
                log_text.insert(tk.END, f"    Walk-Forward Score: {walk_forward_score:.2f}/10.0\n")
                log_text.insert(tk.END, f"    Combined Score: {combined_score:.2f}/10.0 (40% training + 60% walk-forward)\n\n")
            else:
                # Standard mode: show single score
                score = scoring.calculate_backtest_score(result, scoring_config)
                log_text.insert(tk.END, f"  ⭐ Backtest Score: {score:.2f}/10.0\n\n")
        except Exception as e:
            print(f"Error calculating score: {e}")
        
        log_text.insert(tk.END, "  Output Results 1:\n")
        for key, value in result['outputresults1'].items():
            log_text.insert(tk.END, f"    {key}: {value}\n")
        log_text.insert(tk.END, "  Output Results 2:\n")
        for key, value in result['outputresults2'].items():
            log_text.insert(tk.END, f"    {key}: {value}\n")
        log_text.insert(tk.END, "  Parameter Stability Metrics:\n")
        for key, value in result['param_stability'].items():
            log_text.insert(tk.END, f"    {key}: {value}\n")
        log_text.insert(tk.END, "\n")

    # Show interactive chart if enabled and in single stock mode
    mode = mode_var.get()
    if show_visuals_var.get() and mode == "Single Stock" and not cancel_requested:
        single_ticker = list(algorithm_results.keys())[0]
        single_result = algorithm_results[single_ticker]
        if "Error" not in single_result and data is not None:  # Add check for data
            ticker_df = data[data["Ticker"] == single_ticker].copy()
            visual.show_interactive_chart(single_ticker, ticker_df, single_result)

def view_trade_table():
    """Open a new window to display the best trades for selected tickers or cached backtests."""
    from pathlib import Path
    import pickle
    
    trade_window = tk.Toplevel(root)
    trade_window.title("Best Trades Table")
    trade_window.geometry("1200x700")

    # Source selection frame
    source_frame = ttk.Frame(trade_window, padding="10")
    source_frame.pack(fill="x", padx=10, pady=5)
    
    ttk.Label(source_frame, text="Data Source:", font=("Arial", 12)).pack(side="left", padx=5)
    source_var = tk.StringVar(value="Current Results")
    source_radio1 = ttk.Radiobutton(source_frame, text="Current Results", variable=source_var, value="Current Results")
    source_radio1.pack(side="left", padx=10)
    source_radio2 = ttk.Radiobutton(source_frame, text="Cached Backtest", variable=source_var, value="Cached Backtest")
    source_radio2.pack(side="left", padx=10)
    
    # Cache file selection (hidden initially)
    cache_frame = ttk.Frame(trade_window, padding="5")
    cache_frame.pack(fill="x", padx=10, pady=5)
    
    ttk.Label(cache_frame, text="Select Cache File:").pack(side="left", padx=5)
    cache_file_var = tk.StringVar()
    cache_dir = Path("backtest_cache")
    cache_files = list(cache_dir.glob("*.pkl")) if cache_dir.exists() else []
    cache_file_names = [f"{f.name}" for f in cache_files]
    cache_combobox = ttk.Combobox(
        cache_frame, 
        textvariable=cache_file_var, 
        values=cache_file_names, 
        state="readonly", 
        width=60
    )
    cache_combobox.pack(side="left", padx=5, fill="x", expand=True)
    if cache_file_names:
        cache_combobox.current(0)
    
    # Ticker selection frame
    ticker_frame = ttk.Frame(trade_window, padding="10")
    ticker_frame.pack(fill="x", padx=10, pady=5)
    
    ttk.Label(ticker_frame, text="Select Ticker:", font=("Arial", 12)).pack(side="left", padx=5)
    ticker_var = tk.StringVar()
    ticker_combobox = ttk.Combobox(ticker_frame, textvariable=ticker_var, state="readonly", width=20)
    ticker_combobox.pack(side="left", padx=5)

    tree_frame_trade = ttk.Frame(trade_window)
    tree_frame_trade.pack(fill="both", expand=True, padx=10, pady=10)

    tree_scroll_y = ttk.Scrollbar(tree_frame_trade, orient="vertical")
    tree_scroll_y.pack(side="right", fill="y")
    tree_scroll_x = ttk.Scrollbar(tree_frame_trade, orient="horizontal")
    tree_scroll_x.pack(side="bottom", fill="x")

    tree_trade = ttk.Treeview(tree_frame_trade,
                              yscrollcommand=tree_scroll_y.set,
                              xscrollcommand=tree_scroll_x.set)
    tree_trade.pack(fill="both", expand=True)
    tree_scroll_y.config(command=tree_trade.yview)
    tree_scroll_x.config(command=tree_trade.xview)
    
    status_label = ttk.Label(trade_window, text="", foreground="gray")
    status_label.pack(pady=5)

    def update_source():
        """Show/hide cache selection based on source."""
        if source_var.get() == "Cached Backtest":
            cache_frame.pack(fill="x", padx=10, pady=5, before=ticker_frame)
            if cache_file_names:
                load_cache_trades()
            else:
                status_label.config(text="No cached backtests found.")
        else:
            cache_frame.pack_forget()
            # Update ticker list for current results
            if algorithm_results:
                tickers = list(algorithm_results.keys())
                ticker_combobox['values'] = tickers
                if tickers:
                    ticker_var.set(tickers[0])
                    populate_trade_table()
            else:
                ticker_combobox['values'] = []
                status_label.config(text="No current results available. Run the algorithm first.")
    
    def load_cache_trades():
        """Load trades from selected cache file."""
        selection = cache_combobox.current()
        if selection < 0 or not cache_files:
            return
        
        cache_file = cache_files[selection]
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            ticker = cache_data.get('ticker', 'N/A')
            ticker_var.set(ticker)
            ticker_combobox['values'] = [ticker]
            
            # Check if this is walk-forward - display trades by segment
            if cache_data.get("walk_forward_mode", False):
                walk_forward_segment_trades = cache_data.get('walk_forward_segment_trades', [])
                if not walk_forward_segment_trades:
                    status_label.config(text="Walk-Forward Analysis: No segment trades available.")
                    for item in tree_trade.get_children():
                        tree_trade.delete(item)
                    tree_trade["columns"] = ()
                    return
                
                # Display walk-forward trades separated by segment
                all_trades = []
                for seg_data in walk_forward_segment_trades:
                    segment_num = seg_data.get('segment', 0)
                    seg_trades = seg_data.get('trades', [])
                    test_start = seg_data.get('test_start', '')
                    test_end = seg_data.get('test_end', '')
                    train_start = seg_data.get('train_start', '')
                    train_end = seg_data.get('train_end', '')
                    
                    # Format dates
                    if isinstance(test_start, datetime):
                        test_period = f"{test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}"
                    else:
                        test_period = f"{test_start} to {test_end}"
                    
                    if isinstance(train_start, datetime):
                        train_period = f"{train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}"
                    else:
                        train_period = f"{train_start} to {train_end}"
                    
                    if seg_trades:
                        for trade in seg_trades:
                            # Format dates if they're datetime objects
                            if isinstance(trade.get('Date'), datetime):
                                trade_date = trade['Date'].strftime('%Y-%m-%d')
                            else:
                                trade_date = str(trade.get('Date', ''))
                            
                            if isinstance(trade.get('BuyDate'), datetime):
                                buy_date = trade['BuyDate'].strftime('%Y-%m-%d')
                            else:
                                buy_date = str(trade.get('BuyDate', ''))
                            
                            if isinstance(trade.get('SellDate'), datetime):
                                sell_date = trade['SellDate'].strftime('%Y-%m-%d')
                            else:
                                sell_date = str(trade.get('SellDate', ''))
                            
                            all_trades.append({
                                'Segment': f"Segment {segment_num}",
                                'Training Period': train_period,
                                'Walk-Forward Test Period': test_period,
                                'Trade Date': trade_date,
                                'BuyDate': buy_date,
                                'SellDate': sell_date,
                                'BuyPrice': f"{trade.get('BuyPrice', 0):.2f}",
                                'SellPrice': f"{trade.get('SellPrice', 0):.2f}",
                                'PreTaxReturn': f"{trade.get('PreTaxReturn', 0):.4f}",
                                'HoldTime': trade.get('HoldTime', 0),
                                'GainDollars': f"${trade.get('GainDollars', 0):.2f}",
                                'SMA_A': trade.get('SMA_A', ''),
                                'SMA_B': trade.get('SMA_B', '')
                            })
                
                if all_trades:
                    populate_trade_table_from_data(all_trades)
                    total_trades = len(all_trades)
                    segments_count = len(walk_forward_segment_trades)
                    status_label.config(text=f"✓ WALK-FORWARD (TEST) trades: {total_trades} trades across {segments_count} segments for {ticker} | Note: All trades shown are from test periods only")
                else:
                    status_label.config(text="Walk-Forward Analysis: No trades found in segments.")
                    for item in tree_trade.get_children():
                        tree_trade.delete(item)
                    tree_trade["columns"] = ()
                return
            
            # Get best combination
            all_combinations = cache_data.get('all_combinations', [])
            best_idx = cache_data.get('best_combination_idx', 0)
            
            if not all_combinations or best_idx >= len(all_combinations):
                status_label.config(text="No trades available in cached backtest.")
                return
            
            best_combo = all_combinations[best_idx]
            
            # Check if trades are stored in cache
            if 'besttrades' in cache_data and cache_data['besttrades']:
                best_trades = cache_data['besttrades']
                populate_trade_table_from_data(best_trades)
                status_label.config(text=f"Showing trades for {ticker} (SMA: {best_combo.get('sma_a', 'N/A')}, {best_combo.get('sma_b', 'N/A')})")
            else:
                status_label.config(text=f"Trades not stored in cache. Only aggregated metrics are available. (SMA: {best_combo.get('sma_a', 'N/A')}, {best_combo.get('sma_b', 'N/A')})")
                for item in tree_trade.get_children():
                    tree_trade.delete(item)
                tree_trade["columns"] = ()
                
        except Exception as e:
            status_label.config(text=f"Error loading cache: {e}")
            import traceback
            traceback.print_exc()

    def populate_trade_table_from_data(best_trades):
        """Populate trade table from trade data."""
        for item in tree_trade.get_children():
            tree_trade.delete(item)
        tree_trade["columns"] = ()
        tree_trade["show"] = "headings"

        if not best_trades:
            return

        columns = list(best_trades[0].keys())
        tree_trade["columns"] = columns
        for col in columns:
            tree_trade.heading(col, text=col)
            tree_trade.column(col, anchor="center", width=120)

        for trade in best_trades:
            values = [str(trade.get(col, '')) for col in columns]
            tree_trade.insert("", "end", values=values)

    def populate_trade_table(event=None):
        """Populate trade table from current results."""
        if source_var.get() == "Cached Backtest":
            load_cache_trades()
            return
            
        selected_ticker = ticker_var.get()
        if not selected_ticker:
            return

        for item in tree_trade.get_children():
            tree_trade.delete(item)
        tree_trade["columns"] = ()
        tree_trade["show"] = "headings"

        if not algorithm_results or selected_ticker not in algorithm_results:
            status_label.config(text="No data available for selected ticker.")
            return

        result = algorithm_results[selected_ticker]
        if "Error" in result:
            status_label.config(text=f"Error: {result['Error']}")
            return

        # Check if walk-forward - display trades by segment
        if result.get("walk_forward_mode", False):
            walk_forward_segment_trades = result.get('walk_forward_segment_trades', [])
            if not walk_forward_segment_trades:
                status_label.config(text="Walk-Forward Analysis: No segment trades available.")
                return
            
            # Display walk-forward trades separated by segment
            all_trades = []
            for seg_data in walk_forward_segment_trades:
                segment_num = seg_data.get('segment', 0)
                seg_trades = seg_data.get('trades', [])
                test_start = seg_data.get('test_start', '')
                test_end = seg_data.get('test_end', '')
                train_start = seg_data.get('train_start', '')
                train_end = seg_data.get('train_end', '')
                
                # Format dates
                if isinstance(test_start, datetime):
                    test_period = f"{test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}"
                else:
                    test_period = f"{test_start} to {test_end}"
                
                if isinstance(train_start, datetime):
                    train_period = f"{train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}"
                else:
                    train_period = f"{train_start} to {train_end}"
                
                if seg_trades:
                    for trade in seg_trades:
                        # Format dates if they're datetime objects
                        if isinstance(trade.get('Date'), datetime):
                            trade_date = trade['Date'].strftime('%Y-%m-%d')
                        else:
                            trade_date = str(trade.get('Date', ''))
                        
                        if isinstance(trade.get('BuyDate'), datetime):
                            buy_date = trade['BuyDate'].strftime('%Y-%m-%d')
                        else:
                            buy_date = str(trade.get('BuyDate', ''))
                        
                        if isinstance(trade.get('SellDate'), datetime):
                            sell_date = trade['SellDate'].strftime('%Y-%m-%d')
                        else:
                            sell_date = str(trade.get('SellDate', ''))
                        
                        all_trades.append({
                            'Segment': f"Segment {segment_num}",
                            'Training Period': train_period,
                            'Walk-Forward Test Period': test_period,
                            'Trade Date': trade_date,
                            'BuyDate': buy_date,
                            'SellDate': sell_date,
                            'BuyPrice': f"{trade.get('BuyPrice', 0):.2f}",
                            'SellPrice': f"{trade.get('SellPrice', 0):.2f}",
                            'PreTaxReturn': f"{trade.get('PreTaxReturn', 0):.4f}",
                            'HoldTime': trade.get('HoldTime', 0),
                            'GainDollars': f"${trade.get('GainDollars', 0):.2f}",
                            'SMA_A': trade.get('SMA_A', ''),
                            'SMA_B': trade.get('SMA_B', '')
                        })
            
            if all_trades:
                populate_trade_table_from_data(all_trades)
                total_trades = len(all_trades)
                segments_count = len(walk_forward_segment_trades)
                status_label.config(text=f"✓ WALK-FORWARD (TEST) trades: {total_trades} trades across {segments_count} segments for {selected_ticker} | Note: Training period trades are not shown")
            else:
                status_label.config(text="Walk-Forward Analysis: No trades found in segments.")
            return

        best_trades = result.get("besttrades", [])
        if not best_trades:
            status_label.config(text=f"No trades were generated for {selected_ticker}.")
            return

        populate_trade_table_from_data(best_trades)
        status_label.config(text=f"Showing {len(best_trades)} trades for {selected_ticker}")

    # Set up initial state
    source_var.trace("w", lambda *args: update_source())
    cache_combobox.bind("<<ComboboxSelected>>", lambda e: load_cache_trades())
    ticker_combobox.bind("<<ComboboxSelected>>", populate_trade_table)
    
    # Initialize
    if algorithm_results:
        tickers = list(algorithm_results.keys())
        ticker_combobox['values'] = tickers
        if tickers:
            ticker_var.set(tickers[0])
            populate_trade_table()
    else:
        status_label.config(text="No current results available. Select a cached backtest or run the algorithm first.")

def export_results_to_csv():
    """Export the algorithm results to a CSV file with the specified format."""
    if not algorithm_results:
        messagebox.showwarning("No Data", "Please run the algorithm first before exporting results.")
        return

    export_data = []
    today_str = datetime.now().strftime("%Y-%m-%d")
    strategy = "sma2024"
    
    # Get timeframe from new year/month inputs
    if timeframe_all_available_var.get():
        total_timeframe = "All Available"
    else:
        years = timeframe_years_var.get()
        months = timeframe_months_var.get()
        if years > 0 and months > 0:
            total_timeframe = f"{years}Y {months}M"
        elif years > 0:
            total_timeframe = f"{years} Years"
        elif months > 0:
            total_timeframe = f"{months} Months"
        else:
            total_timeframe = "Custom"
    
    # Check if walk-forward mode was used
    is_walk_forward = any(result.get("walk_forward_mode", False) for result in algorithm_results.values() if "Error" not in result)

    for symbol, result in algorithm_results.items():
        if "Error" in result:
            continue

        output1 = result.get("outputresults1", {})
        output2 = result.get("outputresults2", {})
        param_stability = result.get("param_stability", {})
        
        # Determine mode
        mode = "WalkForward" if result.get("walk_forward_mode", False) else "Standard"
        segments = result.get("segments", 0) if result.get("walk_forward_mode", False) else 0
        training_score = result.get("training_score", None) if result.get("walk_forward_mode", False) else None
        walk_forward_score = result.get("walk_forward_score", None) if result.get("walk_forward_mode", False) else None
        combined_score = result.get("combined_score", None) if result.get("walk_forward_mode", False) else None

        besta = output1.get("besta", "")
        bestb = output1.get("bestb", "")
        betteroff = output1.get("betteroff", 0)
        besttaxedreturn = output1.get("besttaxedreturn", 0)
        noalgoreturn = output1.get("noalgoreturn", 0)
        winningtradepct = output2.get("winningtradepct", 0)
        maxdrawdown = output2.get("maxdrawdown(worst trade return pct)", 0)
        besttradecount = output1.get("besttradecount", 0)
        avg_hold_time = output2.get("average_hold_time", 0)
        win_pct_last_4 = output2.get("win_percentage_last_4_trades", None)
        optimization_objective = output1.get("optimization_objective", "taxed_return")

        # Convert internal value to display label for export
        optimization_display = {v: k for k, v in optimization_mapping.items()}.get(optimization_objective, "Taxed Return")

        export_row = {
            "Symbol": symbol,
            "Test Date": today_str,
            "strategy": strategy,
            "Mode": mode,
            "Walk-Forward Segments": segments if mode == "WalkForward" else "",
            "Training Score": training_score if training_score is not None else "",
            "Walk-Forward Score": walk_forward_score if walk_forward_score is not None else "",
            "Combined Score": combined_score if combined_score is not None else "",
            "Total Timeframe": total_timeframe,
            "Optimization Objective": optimization_display,
            "Input 1": besta,
            "Input 2": bestb,
            "5 year diff - %": betteroff,
            "Taxed Return": besttaxedreturn,
            "NoAlgoReturn": noalgoreturn,
            "Win Rate": winningtradepct,
            "Max Drawdown": maxdrawdown,
            "# of Closed Trades": besttradecount,
            "Avg Hold Time": avg_hold_time,
            "Win % Last 4 Trades": win_pct_last_4,
            # Parameter Stability Metrics - Taxed Return
            "Taxed Return Avg": param_stability.get("taxed_return_avg", 0),
            "Taxed Return Std": param_stability.get("taxed_return_std", 0),
            "Taxed Return Max": param_stability.get("taxed_return_max", 0),
            "Taxed Return Min": param_stability.get("taxed_return_min", 0),
            "Taxed Return Max-Min": param_stability.get("taxed_return_max_min_diff", 0),
            "Taxed Return Max-Avg": param_stability.get("taxed_return_max_avg_diff", 0),
            # Parameter Stability Metrics - Better Off
            "Better Off Avg": param_stability.get("better_off_avg", 0),
            "Better Off Std": param_stability.get("better_off_std", 0),
            "Better Off Max": param_stability.get("better_off_max", 0),
            "Better Off Min": param_stability.get("better_off_min", 0),
            "Better Off Max-Min": param_stability.get("better_off_max_min_diff", 0),
            "Better Off Max-Avg": param_stability.get("better_off_max_avg_diff", 0),
            # Parameter Stability Metrics - Win Rate
            "Win Rate Avg": param_stability.get("win_rate_avg", 0),
            "Win Rate Std": param_stability.get("win_rate_std", 0),
            "Win Rate Max": param_stability.get("win_rate_max", 0),
            "Win Rate Min": param_stability.get("win_rate_min", 0),
            "Win Rate Max-Min": param_stability.get("win_rate_max_min_diff", 0),
            "Win Rate Max-Avg": param_stability.get("win_rate_max_avg_diff", 0),
            # Parameter Stability Metrics - Trade Count
            "Trade Count Avg": param_stability.get("trade_count_avg", 0),
            "Trade Count Std": param_stability.get("trade_count_std", 0),
            "Trade Count Max": param_stability.get("trade_count_max", 0),
            "Trade Count Min": param_stability.get("trade_count_min", 0),
            "Trade Count Max-Min": param_stability.get("trade_count_max_min_diff", 0),
            "Trade Count Max-Avg": param_stability.get("trade_count_max_avg_diff", 0)
        }

        export_data.append(export_row)

    export_df = pd.DataFrame(export_data, columns=[
        "Symbol",
        "Test Date",
        "strategy",
        "Mode",
        "Walk-Forward Segments",
        "Training Score",
        "Walk-Forward Score",
        "Combined Score",
        "Total Timeframe",
        "Optimization Objective",
        "Input 1",
        "Input 2",
        "5 year diff - %",
        "Taxed Return",
        "NoAlgoReturn",
        "Win Rate",
        "Max Drawdown",
        "# of Closed Trades",
        "Avg Hold Time",
        "Win % Last 4 Trades",
        # Parameter Stability Metrics - Taxed Return
        "Taxed Return Avg",
        "Taxed Return Std",
        "Taxed Return Max",
        "Taxed Return Min",
        "Taxed Return Max-Min",
        "Taxed Return Max-Avg",
        # Parameter Stability Metrics - Better Off
        "Better Off Avg",
        "Better Off Std",
        "Better Off Max",
        "Better Off Min",
        "Better Off Max-Min",
        "Better Off Max-Avg",
        # Parameter Stability Metrics - Win Rate
        "Win Rate Avg",
        "Win Rate Std",
        "Win Rate Max",
        "Win Rate Min",
        "Win Rate Max-Min",
        "Win Rate Max-Avg",
        # Parameter Stability Metrics - Trade Count
        "Trade Count Avg",
        "Trade Count Std",
        "Trade Count Max",
        "Trade Count Min",
        "Trade Count Max-Min",
        "Trade Count Max-Avg"
    ])

    if export_df.empty:
        messagebox.showinfo("No Data", "There are no results to export.")
        return

    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        title="Save Results As"
    )
    if not file_path:
        return

    try:
        export_df.to_csv(file_path, index=False)
        messagebox.showinfo("Success", f"Results successfully exported to {file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save CSV file: {e}")

def load_batch_from_folder():
    """Load a batch run from a selected batch folder and open batch view."""
    global algorithm_results, scoring_config
    from pathlib import Path
    import pickle
    import cache_manager
    
    # Get batch cache directory
    cache_dir = cache_manager.CACHE_DIR
    
    # Find all batch folders (subdirectories in backtest_cache)
    batch_folders = [d for d in cache_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not batch_folders:
        messagebox.showinfo("No Batch Folders", "No batch folders found in backtest_cache directory.")
        return
    
    # Create selection window
    select_window = tk.Toplevel(root)
    select_window.title("Select Batch Folder")
    select_window.geometry("600x500")
    
    ttk.Label(select_window, text="Select a batch folder to load:", font=("Arial", 12, "bold")).pack(pady=10)
    
    # Listbox for batch folders
    list_frame = ttk.Frame(select_window)
    list_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    scrollbar = ttk.Scrollbar(list_frame)
    scrollbar.pack(side="right", fill="y")
    
    listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=("Arial", 10))
    scrollbar.config(command=listbox.yview)
    listbox.pack(fill="both", expand=True)
    
    # Populate with batch folders (sorted by name, newest first)
    batch_folders_sorted = sorted(batch_folders, key=lambda x: x.name, reverse=True)
    folder_info = []
    for folder in batch_folders_sorted:
        # Count files in folder and get sample tickers
        cache_files = list(folder.glob("*.pkl"))
        file_count = len(cache_files)
        
        # Try to get a few ticker names from cache files
        tickers = []
        for cache_file in cache_files[:5]:  # Sample first 5
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    ticker = cache_data.get('ticker')
                    if ticker:
                        tickers.append(ticker)
            except:
                pass
        
        # Format folder name (try to parse datetime)
        folder_display = folder.name
        try:
            # Try to parse YYYYMMDD_HHMMSS format
            if len(folder.name) == 15 and '_' in folder.name:
                date_part, time_part = folder.name.split('_')
                folder_display = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} {time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
        except:
            pass
        
        ticker_preview = ", ".join(tickers[:3])
        if len(tickers) > 3:
            ticker_preview += f" +{len(tickers)-3} more"
        elif not ticker_preview:
            ticker_preview = "No tickers found"
        
        display_text = f"{folder_display} | {file_count} files | {ticker_preview}"
        listbox.insert("end", display_text)
        folder_info.append((folder, file_count, tickers))
    
    # Buttons
    button_frame = ttk.Frame(select_window)
    button_frame.pack(fill="x", padx=10, pady=10)
    
    def load_selected_batch():
        global algorithm_results
        selection = listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a batch folder.")
            return
        
        selected_folder = folder_info[selection[0]][0]
        select_window.destroy()
        
        # Load all cache files from the batch folder
        cache_files = list(selected_folder.glob("*.pkl"))
        if not cache_files:
            messagebox.showwarning("Empty Folder", f"No cache files found in {selected_folder.name}")
            return
        
        # Load all cache files and convert to algorithm_results format
        loaded_results = {}
        errors = []
        
        for cache_file in cache_files:
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                ticker = cache_data.get('ticker')
                if not ticker:
                    continue
                
                # Reconstruct result structure from cache data
                all_combinations = cache_data.get('all_combinations', [])
                best_idx = cache_data.get('best_combination_idx', 0)
                noalgoreturn = cache_data.get('noalgoreturn', 0)
                
                # Get best combination
                if all_combinations and best_idx < len(all_combinations):
                    best_combo = all_combinations[best_idx]
                else:
                    # Fallback: use first combination or create empty
                    best_combo = all_combinations[0] if all_combinations else {}
                
                # Reconstruct outputresults1
                outputresults1 = {
                    "besta": best_combo.get('sma_a', ''),
                    "bestb": best_combo.get('sma_b', ''),
                    "besttaxedreturn": best_combo.get('taxed_return', 0),
                    "betteroff": best_combo.get('better_off', 0),
                    "besttradecount": best_combo.get('trade_count', 0),
                    "noalgoreturn": noalgoreturn,
                    "optimization_objective": cache_data.get('optimization_objective', 'taxed_return')
                }
                
                # Reconstruct outputresults2
                outputresults2 = {
                    "winningtradepct": best_combo.get('win_rate', 0),
                    "maxdrawdown(worst trade return pct)": best_combo.get('max_drawdown', 0),
                    "average_hold_time": best_combo.get('avg_hold_time', 0),
                    "bestendtaxed_liquidity": best_combo.get('end_taxed_liquidity', 0),
                    "win_percentage_last_4_trades": best_combo.get('win_pct_last_4', None),
                    "losingtrades": best_combo.get('losing_trades', 0),
                    "losingtradepct": 1 - best_combo.get('win_rate', 0) if best_combo.get('win_rate', 0) else 0
                }
                
                # Get param_stability if available (might not be in cache for walk-forward)
                param_stability = cache_data.get('param_stability', {})
                
                # Build result structure
                result = {
                    "outputresults1": outputresults1,
                    "outputresults2": outputresults2,
                    "param_stability": param_stability,
                    "all_combinations": all_combinations,
                    "best_combination_idx": best_idx,
                    "noalgoreturn": noalgoreturn,
                    "besttrades": cache_data.get('besttrades', []),
                    "walk_forward_mode": cache_data.get('walk_forward_mode', False),
                    "segments": cache_data.get('segments', 0),
                    "training_score": cache_data.get('training_score', 0.0),
                    "walk_forward_score": cache_data.get('walk_forward_score', 0.0),
                    "combined_score": cache_data.get('combined_score', 0.0),
                    "training_metrics": cache_data.get('training_metrics', {}),
                    "walk_forward_metrics": cache_data.get('walk_forward_metrics', {}),
                    "walk_forward_segment_trades": cache_data.get('walk_forward_segment_trades', [])  # Include segment trades
                }
                
                loaded_results[ticker] = result
                
            except Exception as e:
                errors.append(f"{cache_file.name}: {str(e)}")
                continue
        
        if not loaded_results:
            messagebox.showerror("Load Error", f"Could not load any cache files from {selected_folder.name}.\n\nErrors:\n" + "\n".join(errors[:5]))
            return
        
        if errors:
            error_msg = f"Loaded {len(loaded_results)} stocks, but encountered {len(errors)} errors:\n\n" + "\n".join(errors[:5])
            if len(errors) > 5:
                error_msg += f"\n... and {len(errors) - 5} more errors"
            messagebox.showwarning("Partial Load", error_msg)
        
        # Set algorithm_results to loaded data
        algorithm_results = loaded_results
        
        # Open batch view
        try:
            view_batch_results()
        finally:
            # Restore original results (or keep loaded if user wants)
            # For now, we'll keep the loaded results so user can work with them
            pass
    
    ttk.Button(button_frame, text="Load Batch", command=load_selected_batch).pack(side="right", padx=5)
    ttk.Button(button_frame, text="Cancel", command=select_window.destroy).pack(side="right", padx=5)
    
    # Double-click to load
    def on_double_click(event):
        load_selected_batch()
    
    listbox.bind("<Double-1>", on_double_click)

def view_batch_results():
    """Enhanced batch results view with comprehensive metrics and drill-down capability."""
    global algorithm_results, scoring_config
    
    if not algorithm_results:
        messagebox.showwarning("No Data", "Please run the algorithm first before viewing batch results.")
        return
    
    # Filter out errors and check if we have multiple stocks
    valid_results = {k: v for k, v in algorithm_results.items() if "Error" not in v}
    if len(valid_results) < 2:
        messagebox.showinfo("Not a Batch Run", "Batch view is only available when multiple stocks are run. Current results show only one stock.")
        return
    
    # Check if any have walk forward mode
    has_walk_forward = any(r.get("walk_forward_mode", False) for r in valid_results.values())
    
    # Create enhanced batch view window
    batch_window = tk.Toplevel(root)
    batch_window.title("Enhanced Batch Results Analysis")
    batch_window.geometry("1800x1000")
    
    # Create notebook for tabs
    notebook = ttk.Notebook(batch_window)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)
    
    # ========== SUMMARY TAB ==========
    summary_tab = ttk.Frame(notebook)
    notebook.add(summary_tab, text="Summary Overview")
    
    # Summary stats frame
    summary_frame = ttk.LabelFrame(summary_tab, text="Batch Summary Statistics", padding="10")
    summary_frame.pack(fill="x", padx=10, pady=10)
    
    # Calculate aggregate statistics
    all_scores = []
    all_returns = []
    all_wf_scores = []
    all_wf_returns = []
    
    for symbol, result in valid_results.items():
        if has_walk_forward and result.get("walk_forward_mode", False):
            all_wf_scores.append(result.get("walk_forward_score", 0.0))
            wf_metrics = result.get("walk_forward_metrics", {})
            all_wf_returns.append(wf_metrics.get("taxed_return", 0))
        else:
            score = scoring.calculate_backtest_score(result, scoring_config)
            all_scores.append(score)
            output1 = result.get("outputresults1", {})
            all_returns.append(output1.get("besttaxedreturn", 0))
    
    # Summary stats display
    stats_grid = ttk.Frame(summary_frame)
    stats_grid.pack(fill="x", pady=5)
    
    try:
        import numpy as np
    except ImportError:
        # Fallback if numpy not available
        class np:
            @staticmethod
            def mean(x):
                return sum(x) / len(x) if x else 0
            @staticmethod
            def max(x):
                return max(x) if x else 0
            @staticmethod
            def min(x):
                return min(x) if x else 0
    
    if has_walk_forward:
        avg_score = np.mean(all_wf_scores) if all_wf_scores else 0
        max_score = np.max(all_wf_scores) if all_wf_scores else 0
        min_score = np.min(all_wf_scores) if all_wf_scores else 0
        avg_return = np.mean(all_wf_returns) if all_wf_returns else 0
        max_return = np.max(all_wf_returns) if all_wf_returns else 0
    else:
        avg_score = np.mean(all_scores) if all_scores else 0
        max_score = np.max(all_scores) if all_scores else 0
        min_score = np.min(all_scores) if all_scores else 0
        avg_return = np.mean(all_returns) if all_returns else 0
        max_return = np.max(all_returns) if all_returns else 0
    
    ttk.Label(stats_grid, text=f"Total Stocks: {len(valid_results)}", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=20, pady=5, sticky="w")
    ttk.Label(stats_grid, text=f"Average Score: {avg_score:.2f}", font=("Arial", 10)).grid(row=0, column=1, padx=20, pady=5, sticky="w")
    ttk.Label(stats_grid, text=f"Best Score: {max_score:.2f}", font=("Arial", 10)).grid(row=0, column=2, padx=20, pady=5, sticky="w")
    ttk.Label(stats_grid, text=f"Worst Score: {min_score:.2f}", font=("Arial", 10)).grid(row=0, column=3, padx=20, pady=5, sticky="w")
    ttk.Label(stats_grid, text=f"Average Return: {avg_return:.2%}", font=("Arial", 10)).grid(row=1, column=0, padx=20, pady=5, sticky="w")
    ttk.Label(stats_grid, text=f"Best Return: {max_return:.2%}", font=("Arial", 10)).grid(row=1, column=1, padx=20, pady=5, sticky="w")
    
    # Action buttons frame
    export_frame = ttk.Frame(summary_frame)
    export_frame.pack(fill="x", pady=10)
    
    # Left side: Scoring controls
    scoring_frame = ttk.Frame(export_frame)
    scoring_frame.pack(side="left", padx=5)
    
    def open_scoring_from_batch():
        """Open scoring config window from batch view."""
        open_scoring_config()
    
    ttk.Button(scoring_frame, text="⚙️ Scoring Config", command=open_scoring_from_batch).pack(side="left", padx=2)
    
    def rescore_batch():
        """Rescore entire batch with current scoring configuration."""
        global algorithm_results, scoring_config
        
        response = messagebox.askyesno(
            "Rescore Batch",
            f"This will rescore all {len(valid_results)} stocks in the batch using the current scoring configuration.\n\n"
            "The scores will be recalculated and the view will refresh.\n\n"
            "Continue?"
        )
        if not response:
            return
        
        # Rescore all results
        rescored_count = 0
        errors = []
        
        for symbol, result in valid_results.items():
            if "Error" in result:
                continue
            
            try:
                # For walk-forward, we need to rescore training and walk-forward separately
                if result.get("walk_forward_mode", False):
                    # Rescore training period using training_metrics
                    training_metrics = result.get("training_metrics", {})
                    if training_metrics:
                        training_result = {
                            "outputresults1": {
                                "besttaxedreturn": training_metrics.get("taxed_return", 0),
                                "betteroff": training_metrics.get("better_off", 0),
                                "besttradecount": training_metrics.get("trade_count", 0),
                                "noalgoreturn": result.get("noalgoreturn", 0)
                            },
                            "outputresults2": {
                                "winningtradepct": training_metrics.get("win_rate", 0),
                                "maxdrawdown(worst trade return pct)": training_metrics.get("max_drawdown", 0),
                                "average_hold_time": training_metrics.get("avg_hold_time", 0)
                            },
                            "param_stability": result.get("param_stability", {})
                        }
                        new_training_score = scoring.calculate_backtest_score(training_result, scoring_config)
                        result["training_score"] = new_training_score
                    
                    # Rescore walk-forward period
                    wf_metrics = result.get("walk_forward_metrics", {})
                    if wf_metrics:
                        wf_result = {
                            "outputresults1": {
                                "besttaxedreturn": wf_metrics.get("taxed_return", 0),
                                "betteroff": 0.0,  # Walk-forward doesn't track better_off
                                "besttradecount": wf_metrics.get("trade_count", 0),
                                "noalgoreturn": 0.0
                            },
                            "outputresults2": {
                                "winningtradepct": wf_metrics.get("win_rate", 0),
                                "maxdrawdown(worst trade return pct)": wf_metrics.get("max_drawdown", 0),
                                "average_hold_time": wf_metrics.get("avg_hold_time", 0)
                            },
                            "param_stability": {}  # Walk-forward doesn't have param stability
                        }
                        new_wf_score = scoring.calculate_backtest_score(wf_result, scoring_config)
                        result["walk_forward_score"] = new_wf_score
                        
                        # Update combined score
                        training_score = result.get("training_score", 0.0)
                        result["combined_score"] = training_score * 0.4 + new_wf_score * 0.6
                else:
                    # Standard backtest - rescore using current config
                    # The score is calculated on-the-fly, but we can mark it as rescored
                    result["_rescore_timestamp"] = datetime.now().isoformat()
                
                rescored_count += 1
            except Exception as e:
                errors.append(f"{symbol}: {str(e)}")
                print(f"Error rescoring {symbol}: {e}")
                continue
        
        # Update algorithm_results
        algorithm_results.update(valid_results)
        
        # Show results
        if errors:
            error_msg = f"Rescored {rescored_count} stocks successfully.\n\n"
            error_msg += f"Encountered {len(errors)} errors:\n" + "\n".join(errors[:5])
            if len(errors) > 5:
                error_msg += f"\n... and {len(errors) - 5} more errors"
            messagebox.showwarning("Rescore Complete with Errors", error_msg)
        else:
            messagebox.showinfo("Rescore Complete", f"Successfully rescored {rescored_count} stocks.\n\nView will refresh.")
        
        # Refresh the view
        batch_window.destroy()
        view_batch_results()  # Reopen with new scores
    
    ttk.Button(scoring_frame, text="🔄 Rescore Batch", command=rescore_batch).pack(side="left", padx=2)
    
    # Right side: Load Previous Batch and Export
    def load_previous_batch_from_view():
        """Load a previous batch from folder and refresh the view."""
        global algorithm_results
        
        # Store reference to current window so we can close it after loading
        current_window = batch_window
        
        # Load batch from folder
        # We'll call load_batch_from_folder but modify it to work from within the batch view
        from pathlib import Path
        import pickle
        import cache_manager
        
        # Get batch cache directory
        cache_dir = cache_manager.CACHE_DIR
        
        # Find all batch folders (subdirectories in backtest_cache)
        batch_folders = [d for d in cache_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        if not batch_folders:
            messagebox.showinfo("No Batch Folders", "No batch folders found in backtest_cache directory.")
            return
        
        # Create selection window
        select_window = tk.Toplevel(batch_window)
        select_window.title("Select Batch Folder")
        select_window.geometry("600x500")
        
        ttk.Label(select_window, text="Select a batch folder to load:", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Listbox for batch folders
        list_frame = ttk.Frame(select_window)
        list_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        
        listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=("Arial", 10))
        scrollbar.config(command=listbox.yview)
        listbox.pack(fill="both", expand=True)
        
        # Populate with batch folders (sorted by name, newest first)
        batch_folders_sorted = sorted(batch_folders, key=lambda x: x.name, reverse=True)
        folder_info = []
        for folder in batch_folders_sorted:
            try:
                # Try to get file count and date from folder name
                cache_files = list(folder.glob("*.pkl"))
                file_count = len(cache_files)
                folder_display = f"{folder.name} ({file_count} files)"
                folder_info.append((folder, folder_display))
                listbox.insert(tk.END, folder_display)
            except:
                folder_info.append((folder, folder.name))
                listbox.insert(tk.END, folder.name)
        
        button_frame = ttk.Frame(select_window)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        def load_selected_batch_from_view():
            selection = listbox.curselection()
            if not selection:
                messagebox.showwarning("No Selection", "Please select a batch folder.")
                return
            
            selected_folder = folder_info[selection[0]][0]
            select_window.destroy()
            
            # Load all cache files from the batch folder
            cache_files = list(selected_folder.glob("*.pkl"))
            if not cache_files:
                messagebox.showwarning("Empty Folder", f"No cache files found in {selected_folder.name}")
                return
            
            # Load all cache files and convert to algorithm_results format
            loaded_results = {}
            errors = []
            
            for cache_file in cache_files:
                try:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    ticker = cache_data.get('ticker')
                    if not ticker:
                        continue
                    
                    # Reconstruct result structure from cache data
                    all_combinations = cache_data.get('all_combinations', [])
                    best_idx = cache_data.get('best_combination_idx', 0)
                    noalgoreturn = cache_data.get('noalgoreturn', 0)
                    
                    # Get best combination
                    if all_combinations and best_idx < len(all_combinations):
                        best_combo = all_combinations[best_idx]
                    else:
                        # Fallback: use first combination or create empty
                        best_combo = all_combinations[0] if all_combinations else {}
                    
                    # Reconstruct outputresults1 from best_combo (like original function does)
                    outputresults1 = {
                        "besta": best_combo.get('sma_a', ''),
                        "bestb": best_combo.get('sma_b', ''),
                        "besttaxedreturn": best_combo.get('taxed_return', 0),
                        "betteroff": best_combo.get('better_off', 0),
                        "besttradecount": best_combo.get('trade_count', 0),
                        "noalgoreturn": noalgoreturn,
                        "optimization_objective": cache_data.get('optimization_objective', 'taxed_return')
                    }
                    
                    # Reconstruct outputresults2 from best_combo (like original function does)
                    outputresults2 = {
                        "winningtradepct": best_combo.get('win_rate', 0),
                        "maxdrawdown(worst trade return pct)": best_combo.get('max_drawdown', 0),
                        "average_hold_time": best_combo.get('avg_hold_time', 0),
                        "bestendtaxed_liquidity": best_combo.get('end_taxed_liquidity', 0),
                        "win_percentage_last_4_trades": best_combo.get('win_pct_last_4', None),
                        "losingtrades": best_combo.get('losing_trades', 0),
                        "losingtradepct": 1 - best_combo.get('win_rate', 0) if best_combo.get('win_rate', 0) else 0
                    }
                    
                    # Get param_stability if available (might not be in cache for walk-forward)
                    param_stability = cache_data.get('param_stability', {})
                    
                    # Build result structure with all walk-forward fields
                    result = {
                        "outputresults1": outputresults1,
                        "outputresults2": outputresults2,
                        "param_stability": param_stability,
                        "all_combinations": all_combinations,
                        "best_combination_idx": best_idx,
                        "noalgoreturn": noalgoreturn,
                        "besttrades": cache_data.get('besttrades', []),
                        "walk_forward_mode": cache_data.get('walk_forward_mode', False),
                        "segments": cache_data.get('segments', 0),
                        "training_score": cache_data.get('training_score', 0.0),
                        "walk_forward_score": cache_data.get('walk_forward_score', 0.0),
                        "combined_score": cache_data.get('combined_score', 0.0),
                        "training_metrics": cache_data.get('training_metrics', {}),
                        "walk_forward_metrics": cache_data.get('walk_forward_metrics', {}),
                        "walk_forward_segment_trades": cache_data.get('walk_forward_segment_trades', [])  # Include segment trades
                    }
                    
                    loaded_results[ticker] = result
                except Exception as e:
                    errors.append(f"{cache_file.name}: {str(e)}")
            
            if errors:
                messagebox.showwarning("Load Warnings", f"Some files had errors:\n\n" + "\n".join(errors[:10]))
            
            if not loaded_results:
                messagebox.showerror("Load Error", "No valid results could be loaded from the selected folder.")
                return
            
            # Set algorithm_results to loaded data
            algorithm_results = loaded_results
            
            # Close current batch window and open new one with loaded data
            current_window.destroy()
            view_batch_results()
        
        ttk.Button(button_frame, text="Load Batch", command=load_selected_batch_from_view).pack(side="right", padx=5)
        ttk.Button(button_frame, text="Cancel", command=select_window.destroy).pack(side="right", padx=5)
        
        # Double-click to load
        def on_double_click(event):
            load_selected_batch_from_view()
        
        listbox.bind("<Double-1>", on_double_click)
    
    ttk.Button(export_frame, text="📁 Load Previous Batch", command=load_previous_batch_from_view).pack(side="right", padx=5)
    ttk.Button(export_frame, text="📊 Export to CSV", command=lambda: export_batch_to_csv(valid_results)).pack(side="right", padx=5)
    
    # ========== MAIN RESULTS TAB ==========
    results_tab = ttk.Frame(notebook)
    notebook.add(results_tab, text="Ranked Results")
    
    # Top controls
    controls_frame = ttk.Frame(results_tab, padding="10")
    controls_frame.pack(fill="x", padx=10, pady=5)
    
    ttk.Label(controls_frame, text=f"Ranked by: {'Walk-Forward Score' if has_walk_forward else 'Backtest Score'}", 
              font=("Arial", 10, "bold")).pack(side="left", padx=5)
    
    # Filter frame
    filter_frame = ttk.Frame(controls_frame)
    filter_frame.pack(side="right", padx=5)
    ttk.Label(filter_frame, text="Filter:").pack(side="left", padx=2)
    filter_var = tk.StringVar()
    filter_entry = ttk.Entry(filter_frame, textvariable=filter_var, width=20)
    filter_entry.pack(side="left", padx=2)
    
    # Tree frame with scrollbars
    tree_container = ttk.Frame(results_tab)
    tree_container.pack(fill="both", expand=True, padx=10, pady=10)
    
    tree_scroll_y = ttk.Scrollbar(tree_container, orient="vertical")
    tree_scroll_y.pack(side="right", fill="y")
    tree_scroll_x = ttk.Scrollbar(tree_container, orient="horizontal")
    tree_scroll_x.pack(side="bottom", fill="x")
    
    # Define comprehensive columns
    if has_walk_forward:
        columns = ("Rank", "Symbol", "Train_Score", "WF_Score", "Train_Return", "WF_Return", 
                  "Train_WinRate", "WF_WinRate", "Train_Trades", "WF_Trades", "Train_DD", "WF_DD",
                  "Train_HoldTime", "WF_HoldTime", "SMA_A", "SMA_B", "Segments")
        column_names = ("Rank", "Symbol", "Train Score", "WF Score", "Train Return", "WF Return",
                       "Train WR", "WF WR", "Train #", "WF #", "Train DD", "WF DD",
                       "Train Hold", "WF Hold", "SMA A", "SMA B", "Segments")
        column_widths = (60, 80, 90, 90, 100, 100, 80, 80, 70, 70, 80, 80, 80, 80, 60, 60, 70)
    else:
        columns = ("Rank", "Symbol", "Score", "Return", "Better_Off", "Win_Rate", "Trades", 
                  "Wins", "Losses", "Max_DD", "Avg_Hold", "Avg_Trade", "End_Liquidity", "SMA_A", "SMA_B")
        column_names = ("Rank", "Symbol", "Score", "Return", "Better Off", "Win Rate", "Trades",
                       "Wins", "Losses", "Max DD", "Avg Hold", "Avg Trade", "End Liquidity", "SMA A", "SMA B")
        column_widths = (60, 80, 90, 100, 90, 80, 70, 60, 60, 80, 80, 90, 110, 60, 60)
    
    tree_batch = ttk.Treeview(tree_container, columns=columns, show="headings", 
                              yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set,
                              selectmode="browse")
    tree_scroll_y.config(command=tree_batch.yview)
    tree_scroll_x.config(command=tree_batch.xview)
    
    # Configure headings and columns
    for col, name, width in zip(columns, column_names, column_widths):
        tree_batch.heading(col, text=name)
        tree_batch.column(col, width=width, anchor="center")
    
    tree_batch.pack(fill="both", expand=True)
    
    # Configure tags for highlighting
    tree_batch.tag_configure("top_10", background="#e8f5e9")
    tree_batch.tag_configure("top_3", background="#c8e6c9")
    tree_batch.tag_configure("selected", background="#bbdefb")
    
    # Prepare comprehensive data
    batch_data = []
    for symbol, result in valid_results.items():
        data_entry = {"symbol": symbol, "result": result}
        
        if has_walk_forward and result.get("walk_forward_mode", False):
            training_metrics = result.get("training_metrics", {})
            wf_metrics = result.get("walk_forward_metrics", {})
            training_score = result.get("training_score", 0.0)
            wf_score = result.get("walk_forward_score", 0.0)
            segments = result.get("segments", 0)
            
            output1 = result.get("outputresults1", {})
            best_a = output1.get("besta", "")
            best_b = output1.get("bestb", "")
            
            data_entry.update({
                "training_score": training_score,
                "wf_score": wf_score,
                "training_return": training_metrics.get("taxed_return", 0),
                "wf_return": wf_metrics.get("taxed_return", 0),
                "training_winrate": training_metrics.get("win_rate", 0),
                "wf_winrate": wf_metrics.get("win_rate", 0),
                "training_trades": training_metrics.get("trade_count", 0),
                "wf_trades": wf_metrics.get("trade_count", 0),
                "training_dd": training_metrics.get("max_drawdown", 0),
                "wf_dd": wf_metrics.get("max_drawdown", 0),
                "training_holdtime": training_metrics.get("avg_hold_time", 0),
                "wf_holdtime": wf_metrics.get("avg_hold_time", 0),
                "best_a": best_a,
                "best_b": best_b,
                "segments": segments,
                "sort_key": wf_score
            })
        else:
            score = scoring.calculate_backtest_score(result, scoring_config)
            output1 = result.get("outputresults1", {})
            output2 = result.get("outputresults2", {})
            param_stability = result.get("param_stability", {})
            
            data_entry.update({
                "score": score,
                "taxed_return": output1.get("besttaxedreturn", 0),
                "better_off": output1.get("betteroff", 0),
                "win_rate": output2.get("winningtradepct", 0),
                "trade_count": output1.get("besttradecount", 0),
                "winning_trades": output2.get("winningtradepct", 0) * output1.get("besttradecount", 0) if output1.get("besttradecount", 0) > 0 else 0,
                "losing_trades": output1.get("besttradecount", 0) - (output2.get("winningtradepct", 0) * output1.get("besttradecount", 0)) if output1.get("besttradecount", 0) > 0 else 0,
                "max_drawdown": output2.get("maxdrawdown(worst trade return pct)", 0),
                "avg_hold_time": output2.get("average_hold_time", 0),
                "avg_trade_return": output2.get("avgtradepct", 0) if "avgtradepct" in output2 else 0,
                "end_liquidity": output2.get("bestendtaxed_liquidity", 0),
                "best_a": output1.get("besta", ""),
                "best_b": output1.get("bestb", ""),
                "sort_key": score
            })
        
        batch_data.append(data_entry)
    
    # Sort by sort_key
    batch_data.sort(key=lambda x: x["sort_key"], reverse=True)
    
    # Store batch_data for drill-down
    batch_data_store = batch_data.copy()
    
    def populate_tree(data_list=None):
        """Populate tree with data, optionally filtered."""
        if data_list is None:
            data_list = batch_data
        
        # Clear existing items
        for item in tree_batch.get_children():
            tree_batch.delete(item)
        
        # Filter if needed
        filter_text = filter_var.get().upper()
        if filter_text:
            data_list = [d for d in data_list if filter_text in d["symbol"].upper()]
        
        # Populate tree
        for rank, data in enumerate(data_list, 1):
            if has_walk_forward:
                values = (
                    rank,
                    data["symbol"],
                    f"{data.get('training_score', 0):.2f}",
                    f"{data.get('wf_score', 0):.2f}",
                    f"{data.get('training_return', 0):.2%}",
                    f"{data.get('wf_return', 0):.2%}",
                    f"{data.get('training_winrate', 0):.2%}",
                    f"{data.get('wf_winrate', 0):.2%}",
                    data.get("training_trades", 0),
                    data.get("wf_trades", 0),
                    f"{data.get('training_dd', 0):.2%}",
                    f"{data.get('wf_dd', 0):.2%}",
                    f"{data.get('training_holdtime', 0):.1f}",
                    f"{data.get('wf_holdtime', 0):.1f}",
                    data.get("best_a", ""),
                    data.get("best_b", ""),
                    data.get("segments", 0)
                )
            else:
                values = (
                    rank,
                    data["symbol"],
                    f"{data.get('score', 0):.2f}",
                    f"{data.get('taxed_return', 0):.2%}",
                    f"{data.get('better_off', 0):.2%}",
                    f"{data.get('win_rate', 0):.2%}",
                    data.get("trade_count", 0),
                    int(data.get("winning_trades", 0)),
                    int(data.get("losing_trades", 0)),
                    f"{data.get('max_drawdown', 0):.2%}",
                    f"{data.get('avg_hold_time', 0):.1f}",
                    f"{data.get('avg_trade_return', 0):.2%}",
                    f"${data.get('end_liquidity', 0):,.2f}",
                    data.get("best_a", ""),
                    data.get("best_b", "")
                )
            
            item = tree_batch.insert("", "end", values=values, tags=())
            if rank <= 3:
                tree_batch.set(item, "Rank", f"🏆 {rank}")
                tree_batch.item(item, tags=("top_3",))
            elif rank <= 10:
                tree_batch.item(item, tags=("top_10",))
    
    # Initial population
    populate_tree()
    
    # Filter callback
    def on_filter_change(*args):
        populate_tree()
    
    filter_var.trace("w", on_filter_change)
    
    # Double-click to drill down
    def on_double_click(event):
        selection = tree_batch.selection()
        if not selection:
            return
        item = selection[0]
        values = tree_batch.item(item, "values")
        if values:
            symbol = values[1]  # Symbol is in column 1
            # Find the data for this symbol
            stock_data = next((d for d in batch_data_store if d["symbol"] == symbol), None)
            if stock_data:
                show_stock_detail(batch_window, stock_data, has_walk_forward)
    
    tree_batch.bind("<Double-1>", on_double_click)
    
    # Status label
    status_label = ttk.Label(results_tab, text=f"Double-click a row to view detailed analysis | Showing {len(batch_data)} stocks")
    status_label.pack(pady=5)
    
    # ========== DETAIL VIEW FUNCTION ==========
    def show_stock_detail(parent_window, stock_data, has_wf):
        """Show detailed view for a single stock."""
        detail_window = tk.Toplevel(parent_window)
        detail_window.title(f"Detailed Analysis: {stock_data['symbol']}")
        detail_window.geometry("1600x900")
        
        # Create notebook for detail tabs
        detail_notebook = ttk.Notebook(detail_window)
        detail_notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        result = stock_data["result"]
        
        # Overview tab
        overview_tab = ttk.Frame(detail_notebook)
        detail_notebook.add(overview_tab, text="Overview")
        
        # Key metrics frame
        metrics_frame = ttk.LabelFrame(overview_tab, text="Key Performance Metrics", padding="15")
        metrics_frame.pack(fill="x", padx=10, pady=10)
        
        metrics_grid = ttk.Frame(metrics_frame)
        metrics_grid.pack(fill="both", expand=True)
        
        if has_wf and result.get("walk_forward_mode", False):
            training_metrics = result.get("training_metrics", {})
            wf_metrics = result.get("walk_forward_metrics", {})
            
            # Training metrics
            ttk.Label(metrics_grid, text="TRAINING PERIOD", font=("Arial", 11, "bold")).grid(row=0, column=0, columnspan=3, pady=10, sticky="w")
            ttk.Label(metrics_grid, text=f"Score: {result.get('training_score', 0):.2f}/10.0").grid(row=1, column=0, padx=10, pady=5, sticky="w")
            ttk.Label(metrics_grid, text=f"Return: {training_metrics.get('taxed_return', 0):.2%}").grid(row=1, column=1, padx=10, pady=5, sticky="w")
            ttk.Label(metrics_grid, text=f"Win Rate: {training_metrics.get('win_rate', 0):.2%}").grid(row=1, column=2, padx=10, pady=5, sticky="w")
            ttk.Label(metrics_grid, text=f"Trades: {training_metrics.get('trade_count', 0)}").grid(row=2, column=0, padx=10, pady=5, sticky="w")
            ttk.Label(metrics_grid, text=f"Max DD: {training_metrics.get('max_drawdown', 0):.2%}").grid(row=2, column=1, padx=10, pady=5, sticky="w")
            ttk.Label(metrics_grid, text=f"Avg Hold: {training_metrics.get('avg_hold_time', 0):.1f} days").grid(row=2, column=2, padx=10, pady=5, sticky="w")
            
            # Walk-forward metrics
            ttk.Label(metrics_grid, text="WALK-FORWARD PERIOD", font=("Arial", 11, "bold")).grid(row=3, column=0, columnspan=3, pady=(20, 10), sticky="w")
            ttk.Label(metrics_grid, text=f"Score: {result.get('walk_forward_score', 0):.2f}/10.0").grid(row=4, column=0, padx=10, pady=5, sticky="w")
            ttk.Label(metrics_grid, text=f"Return: {wf_metrics.get('taxed_return', 0):.2%}").grid(row=4, column=1, padx=10, pady=5, sticky="w")
            ttk.Label(metrics_grid, text=f"Win Rate: {wf_metrics.get('win_rate', 0):.2%}").grid(row=4, column=2, padx=10, pady=5, sticky="w")
            ttk.Label(metrics_grid, text=f"Trades: {wf_metrics.get('trade_count', 0)}").grid(row=5, column=0, padx=10, pady=5, sticky="w")
            ttk.Label(metrics_grid, text=f"Wins: {wf_metrics.get('winning_trades', 0)}").grid(row=5, column=1, padx=10, pady=5, sticky="w")
            ttk.Label(metrics_grid, text=f"Losses: {wf_metrics.get('losing_trades', 0)}").grid(row=5, column=2, padx=10, pady=5, sticky="w")
            ttk.Label(metrics_grid, text=f"Max DD: {wf_metrics.get('max_drawdown', 0):.2%}").grid(row=6, column=0, padx=10, pady=5, sticky="w")
            ttk.Label(metrics_grid, text=f"Avg Hold: {wf_metrics.get('avg_hold_time', 0):.1f} days").grid(row=6, column=1, padx=10, pady=5, sticky="w")
            ttk.Label(metrics_grid, text=f"Segments: {result.get('segments', 0)}").grid(row=6, column=2, padx=10, pady=5, sticky="w")
        else:
            output1 = result.get("outputresults1", {})
            output2 = result.get("outputresults2", {})
            score = scoring.calculate_backtest_score(result, scoring_config)
            
            ttk.Label(metrics_grid, text=f"Backtest Score: {score:.2f}/10.0", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=3, pady=10, sticky="w")
            ttk.Label(metrics_grid, text=f"Taxed Return: {output1.get('besttaxedreturn', 0):.2%}").grid(row=1, column=0, padx=10, pady=5, sticky="w")
            ttk.Label(metrics_grid, text=f"Better Off: {output1.get('betteroff', 0):.2%}").grid(row=1, column=1, padx=10, pady=5, sticky="w")
            ttk.Label(metrics_grid, text=f"Win Rate: {output2.get('winningtradepct', 0):.2%}").grid(row=1, column=2, padx=10, pady=5, sticky="w")
            ttk.Label(metrics_grid, text=f"Trades: {output1.get('besttradecount', 0)}").grid(row=2, column=0, padx=10, pady=5, sticky="w")
            ttk.Label(metrics_grid, text=f"Max Drawdown: {output2.get('maxdrawdown(worst trade return pct)', 0):.2%}").grid(row=2, column=1, padx=10, pady=5, sticky="w")
            ttk.Label(metrics_grid, text=f"Avg Hold Time: {output2.get('average_hold_time', 0):.1f} days").grid(row=2, column=2, padx=10, pady=5, sticky="w")
            ttk.Label(metrics_grid, text=f"Best SMA: {output1.get('besta', '')}/{output1.get('bestb', '')}").grid(row=3, column=0, padx=10, pady=5, sticky="w")
            ttk.Label(metrics_grid, text=f"End Liquidity: ${output2.get('bestendtaxed_liquidity', 0):,.2f}").grid(row=3, column=1, padx=10, pady=5, sticky="w")
        
        # Combinations tab (similar to view_cached_backtests)
        combos_tab = ttk.Frame(detail_notebook)
        detail_notebook.add(combos_tab, text="All Combinations")
        
        # Tree for combinations
        combos_tree_frame = ttk.Frame(combos_tab)
        combos_tree_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        combos_scroll_y = ttk.Scrollbar(combos_tree_frame, orient="vertical")
        combos_scroll_y.pack(side="right", fill="y")
        combos_scroll_x = ttk.Scrollbar(combos_tree_frame, orient="horizontal")
        combos_scroll_x.pack(side="bottom", fill="x")
        
        combos_columns = ("Backtest_Score", "SMA_A", "SMA_B", "Taxed_Return", "Better_Off", "Win_Rate", 
                         "Trade_Count", "Winning_Trades", "Losing_Trades", "Max_Drawdown", "Avg_Hold_Time", 
                         "Avg_Trade_Return", "Return_Std", "End_Liquidity", "Under1Y_PL", "Over1Y_PL", "Win_Pct_Last4")
        combos_tree = ttk.Treeview(combos_tree_frame, columns=combos_columns, show="headings",
                                   yscrollcommand=combos_scroll_y.set, xscrollcommand=combos_scroll_x.set)
        combos_scroll_y.config(command=combos_tree.yview)
        combos_scroll_x.config(command=combos_tree.xview)
        
        # Configure columns
        combos_tree.heading("Backtest_Score", text="Score")
        combos_tree.heading("SMA_A", text="SMA A")
        combos_tree.heading("SMA_B", text="SMA B")
        combos_tree.heading("Taxed_Return", text="Return")
        combos_tree.heading("Better_Off", text="Better Off")
        combos_tree.heading("Win_Rate", text="Win Rate")
        combos_tree.heading("Trade_Count", text="Trades")
        combos_tree.heading("Winning_Trades", text="Wins")
        combos_tree.heading("Losing_Trades", text="Losses")
        combos_tree.heading("Max_Drawdown", text="Max DD")
        combos_tree.heading("Avg_Hold_Time", text="Avg Hold")
        combos_tree.heading("Avg_Trade_Return", text="Avg Trade")
        combos_tree.heading("Return_Std", text="Return Std")
        combos_tree.heading("End_Liquidity", text="End Liquidity")
        combos_tree.heading("Under1Y_PL", text="Under 1Y P/L")
        combos_tree.heading("Over1Y_PL", text="Over 1Y P/L")
        combos_tree.heading("Win_Pct_Last4", text="Win % Last 4")
        
        # Set column widths
        for col in combos_columns:
            combos_tree.column(col, width=100, anchor="center")
        
        combos_tree.pack(fill="both", expand=True)
        
        # Populate combinations
        all_combinations = result.get("all_combinations", [])
        best_idx = result.get("best_combination_idx", 0)
        noalgoreturn = result.get("noalgoreturn", 0)
        param_stability = result.get("param_stability", {})
        
        if all_combinations:
            # Score and sort combinations
            combo_scores = []
            for combo in all_combinations:
                combo_result = {
                    "outputresults1": {
                        "besttaxedreturn": combo.get("taxed_return", 0),
                        "betteroff": combo.get("better_off", 0),
                        "besttradecount": combo.get("trade_count", 0),
                        "noalgoreturn": noalgoreturn
                    },
                    "outputresults2": {
                        "winningtradepct": combo.get("win_rate", 0),
                        "maxdrawdown(worst trade return pct)": combo.get("max_drawdown", 0),
                        "average_hold_time": combo.get("avg_hold_time", 0)
                    },
                    "param_stability": param_stability
                }
                backtest_score = scoring.calculate_backtest_score(combo_result, scoring_config)
                combo_scores.append((combo, backtest_score))
            
            # Sort by score
            combo_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Populate tree
            for i, (combo, score) in enumerate(combo_scores):
                win_pct_last4 = combo.get("win_pct_last_4", None)
                win_pct_last4_str = f"{win_pct_last4:.2%}" if win_pct_last4 is not None else "N/A"
                
                values = (
                    f"{score:.2f}",
                    combo.get('sma_a', ''),
                    combo.get('sma_b', ''),
                    f"{combo.get('taxed_return', 0):.4%}",
                    f"{combo.get('better_off', 0):.4%}",
                    f"{combo.get('win_rate', 0):.2%}",
                    combo.get('trade_count', 0),
                    combo.get('winning_trades', 0),
                    combo.get('losing_trades', 0),
                    f"{combo.get('max_drawdown', 0):.4%}",
                    f"{combo.get('avg_hold_time', 0):.1f}",
                    f"{combo.get('avg_trade_return', 0):.4%}",
                    f"{combo.get('return_std', 0):.4f}",
                    f"${combo.get('end_taxed_liquidity', 0):,.2f}",
                    f"${combo.get('under1yearpl', 0):,.2f}",
                    f"${combo.get('over1yearpl', 0):,.2f}",
                    win_pct_last4_str
                )
                
                tags = []
                if combo == all_combinations[best_idx]:
                    tags.append("best")
                if i == 0:  # Highest score
                    tags.append("top_score")
                
                combos_tree.insert("", "end", values=values, tags=tags)
            
            # Configure tags
            combos_tree.tag_configure("best", background="#fff9c4")
            combos_tree.tag_configure("top_score", background="#c8e6c9")
        else:
            ttk.Label(combos_tab, text="No combinations data available", font=("Arial", 12)).pack(pady=20)
        
        # Parameter stability tab
        stability_tab = ttk.Frame(detail_notebook)
        detail_notebook.add(stability_tab, text="Parameter Stability")
        
        param_stability = result.get("param_stability", {})
        if param_stability:
            stability_text = tk.Text(stability_tab, wrap="word", padx=20, pady=20)
            stability_text.pack(fill="both", expand=True)
            stability_text.insert("1.0", "Parameter Stability Metrics:\n\n")
            stability_text.insert("end", f"Taxed Return Std Dev: {param_stability.get('taxed_return_std', 0):.4f}\n")
            stability_text.insert("end", f"Better Off Std Dev: {param_stability.get('better_off_std', 0):.4f}\n")
            stability_text.insert("end", f"Win Rate Std Dev: {param_stability.get('win_rate_std', 0):.4f}\n")
            stability_text.insert("end", f"Taxed Return Max-Min Diff: {param_stability.get('taxed_return_max_min_diff', 0):.4f}\n")
            stability_text.config(state="disabled")
        else:
            ttk.Label(stability_tab, text="No parameter stability data available").pack(pady=20)
        
        # Trades tab - show walk-forward trades if available
        trades_tab = ttk.Frame(detail_notebook)
        detail_notebook.add(trades_tab, text="Trades")
        
        # Explanation label for walk-forward
        trades_explanation_detail = ttk.Label(trades_tab, text="", foreground="blue", font=("Arial", 9), wraplength=800)
        trades_explanation_detail.pack(pady=5, padx=5)
        
        trades_tree_frame = ttk.Frame(trades_tab)
        trades_tree_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        trades_scroll_y = ttk.Scrollbar(trades_tree_frame, orient="vertical")
        trades_scroll_y.pack(side="right", fill="y")
        trades_scroll_x = ttk.Scrollbar(trades_tree_frame, orient="horizontal")
        trades_scroll_x.pack(side="bottom", fill="x")
        
        trades_detail_tree = ttk.Treeview(trades_tree_frame, yscrollcommand=trades_scroll_y.set,
                                          xscrollcommand=trades_scroll_x.set)
        trades_scroll_y.config(command=trades_detail_tree.yview)
        trades_scroll_x.config(command=trades_detail_tree.xview)
        trades_detail_tree.pack(fill="both", expand=True)
        
        trades_status_label = ttk.Label(trades_tab, text="")
        trades_status_label.pack(pady=5)
        
        # Populate trades
        if has_wf and result.get("walk_forward_mode", False):
            walk_forward_segment_trades = result.get('walk_forward_segment_trades', [])
            if walk_forward_segment_trades:
                # Show explanation
                explanation_text = (
                    "WALK-FORWARD ANALYSIS EXPLANATION:\n"
                    "• Each segment has a TRAINING period (to find best parameters) and a TEST period (to test those parameters)\n"
                    "• ALL trades shown below are from the WALK-FORWARD (TEST) periods only\n"
                    "• Training period trades are NOT shown (they're only used for optimization)\n"
                    "• The 'Walk-Forward Test Period' column shows when these trades occurred"
                )
                trades_explanation_detail.config(text=explanation_text)
                
                all_trades = []
                for seg_data in walk_forward_segment_trades:
                    segment_num = seg_data.get('segment', 0)
                    seg_trades = seg_data.get('trades', [])
                    test_start = seg_data.get('test_start', '')
                    test_end = seg_data.get('test_end', '')
                    train_start = seg_data.get('train_start', '')
                    train_end = seg_data.get('train_end', '')
                    
                    # Format dates
                    if isinstance(test_start, datetime):
                        test_period = f"{test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}"
                    else:
                        test_period = f"{test_start} to {test_end}"
                    
                    if isinstance(train_start, datetime):
                        train_period = f"{train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}"
                    else:
                        train_period = f"{train_start} to {train_end}"
                    
                    if seg_trades:
                        for trade in seg_trades:
                            # Format dates
                            if isinstance(trade.get('Date'), datetime):
                                trade_date = trade['Date'].strftime('%Y-%m-%d')
                            else:
                                trade_date = str(trade.get('Date', ''))
                            
                            if isinstance(trade.get('BuyDate'), datetime):
                                buy_date = trade['BuyDate'].strftime('%Y-%m-%d')
                            else:
                                buy_date = str(trade.get('BuyDate', ''))
                            
                            if isinstance(trade.get('SellDate'), datetime):
                                sell_date = trade['SellDate'].strftime('%Y-%m-%d')
                            else:
                                sell_date = str(trade.get('SellDate', ''))
                            
                            all_trades.append({
                                'Segment': f"Segment {segment_num}",
                                'Training Period': train_period,
                                'Walk-Forward Test Period': test_period,
                                'Trade Date': trade_date,
                                'BuyDate': buy_date,
                                'SellDate': sell_date,
                                'BuyPrice': f"{trade.get('BuyPrice', 0):.2f}",
                                'SellPrice': f"{trade.get('SellPrice', 0):.2f}",
                                'PreTaxReturn': f"{trade.get('PreTaxReturn', 0):.4f}",
                                'HoldTime': trade.get('HoldTime', 0),
                                'GainDollars': f"${trade.get('GainDollars', 0):.2f}",
                                'SMA_A': trade.get('SMA_A', ''),
                                'SMA_B': trade.get('SMA_B', '')
                            })
                
                if all_trades:
                    columns = ['Segment', 'Training Period', 'Walk-Forward Test Period', 'Trade Date', 'BuyDate', 'SellDate', 
                              'BuyPrice', 'SellPrice', 'PreTaxReturn', 'HoldTime', 'GainDollars', 'SMA_A', 'SMA_B']
                    trades_detail_tree["columns"] = columns
                    for col in columns:
                        trades_detail_tree.heading(col, text=col)
                        trades_detail_tree.column(col, anchor="center", width=100)
                    
                    for trade in all_trades:
                        values = [str(trade.get(col, '')) for col in columns]
                        trades_detail_tree.insert("", "end", values=values)
                    
                    total_trades = len(all_trades)
                    segments_count = len(walk_forward_segment_trades)
                    trades_status_label.config(text=f"✓ Showing {total_trades} WALK-FORWARD (TEST) trades across {segments_count} segments")
                else:
                    trades_explanation_detail.config(text="")
                    trades_status_label.config(text="Walk-Forward Analysis: No trades found in segments.")
            else:
                trades_explanation_detail.config(text="")
                trades_status_label.config(text="Walk-Forward Analysis: No segment trades available.")
        else:
            # Regular backtest - show besttrades if available
            best_trades = result.get("besttrades", [])
            if best_trades:
                columns = list(best_trades[0].keys()) if best_trades else []
                trades_detail_tree["columns"] = columns
                for col in columns:
                    trades_detail_tree.heading(col, text=col)
                    trades_detail_tree.column(col, anchor="center", width=100)
                
                for trade in best_trades:
                    values = [str(trade.get(col, '')) for col in columns]
                    trades_detail_tree.insert("", "end", values=values)
                
                trades_status_label.config(text=f"Showing {len(best_trades)} trades")
            else:
                trades_status_label.config(text="No trades available for this backtest.")

def export_batch_to_csv(valid_results):
    """Export batch results to CSV with training and walk-forward metrics."""
    global scoring_config
    
    if not valid_results:
        messagebox.showwarning("No Data", "No valid results to export.")
        return
    
    has_walk_forward = any(r.get("walk_forward_mode", False) for r in valid_results.values())
    
    export_data = []
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    for symbol, result in valid_results.items():
        if has_walk_forward and result.get("walk_forward_mode", False):
            training_metrics = result.get("training_metrics", {})
            wf_metrics = result.get("walk_forward_metrics", {})
            training_score = result.get("training_score", 0.0)
            wf_score = result.get("walk_forward_score", 0.0)
            
            output1 = result.get("outputresults1", {})
            best_a = output1.get("besta", "")
            best_b = output1.get("bestb", "")
            
            export_row = {
                "Symbol": symbol,
                "Export Date": today_str,
                "Training Score": training_score,
                "Walk-Forward Score": wf_score,
                "Training Taxed Return": training_metrics.get("taxed_return", 0),
                "Training Win Rate": training_metrics.get("win_rate", 0),
                "Training Trade Count": training_metrics.get("trade_count", 0),
                "Training Max Drawdown": training_metrics.get("max_drawdown", 0),
                "Training Avg Hold Time": training_metrics.get("avg_hold_time", 0),
                "Walk-Forward Taxed Return": wf_metrics.get("taxed_return", 0),
                "Walk-Forward Win Rate": wf_metrics.get("win_rate", 0),
                "Walk-Forward Trade Count": wf_metrics.get("trade_count", 0),
                "Walk-Forward Winning Trades": wf_metrics.get("winning_trades", 0),
                "Walk-Forward Losing Trades": wf_metrics.get("losing_trades", 0),
                "Walk-Forward Max Drawdown": wf_metrics.get("max_drawdown", 0),
                "Walk-Forward Avg Hold Time": wf_metrics.get("avg_hold_time", 0),
                "Best SMA A": best_a,
                "Best SMA B": best_b
            }
        else:
            score = scoring.calculate_backtest_score(result, scoring_config)
            output1 = result.get("outputresults1", {})
            output2 = result.get("outputresults2", {})
            
            export_row = {
                "Symbol": symbol,
                "Export Date": today_str,
                "Backtest Score": score,
                "Taxed Return": output1.get("besttaxedreturn", 0),
                "Win Rate": output2.get("winningtradepct", 0),
                "Trade Count": output1.get("besttradecount", 0),
                "Max Drawdown": output2.get("maxdrawdown(worst trade return pct)", 0),
                "Avg Hold Time": output2.get("average_hold_time", 0),
                "Best SMA A": output1.get("besta", ""),
                "Best SMA B": output1.get("bestb", "")
            }
        
        export_data.append(export_row)
    
    # Sort by walk forward score or backtest score
    if has_walk_forward:
        export_data.sort(key=lambda x: x.get("Walk-Forward Score", 0), reverse=True)
    else:
        export_data.sort(key=lambda x: x.get("Backtest Score", 0), reverse=True)
    
    # Add rank
    for i, row in enumerate(export_data, 1):
        row["Rank"] = i
    
    # Reorder columns to put Rank first
    if has_walk_forward:
        columns = ["Rank", "Symbol", "Export Date", "Training Score", "Walk-Forward Score",
                  "Training Taxed Return", "Walk-Forward Taxed Return",
                  "Training Win Rate", "Walk-Forward Win Rate",
                  "Training Trade Count", "Walk-Forward Trade Count",
                  "Training Max Drawdown", "Walk-Forward Max Drawdown",
                  "Training Avg Hold Time", "Walk-Forward Avg Hold Time",
                  "Walk-Forward Winning Trades", "Walk-Forward Losing Trades",
                  "Best SMA A", "Best SMA B"]
    else:
        columns = ["Rank", "Symbol", "Export Date", "Backtest Score", "Taxed Return",
                  "Win Rate", "Trade Count", "Max Drawdown", "Avg Hold Time",
                  "Best SMA A", "Best SMA B"]
    
    export_df = pd.DataFrame(export_data, columns=columns)
    
    if export_df.empty:
        messagebox.showinfo("No Data", "There are no results to export.")
        return
    
    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        title="Save Batch Results As"
    )
    if not file_path:
        return
    
    try:
        export_df.to_csv(file_path, index=False)
        messagebox.showinfo("Success", f"Batch results successfully exported to {file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save CSV file: {e}")

def open_scoring_config():
    """Open a window to configure scoring parameters."""
    global scoring_config
    
    config_window = tk.Toplevel(root)
    config_window.title("Scoring Configuration")
    config_window.geometry("1000x800")
    
    # Create notebook for tabs (weights and thresholds)
    notebook = ttk.Notebook(config_window)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Weights Tab
    weights_frame = ttk.Frame(notebook, padding="10")
    notebook.add(weights_frame, text="Metric Weights")
    
    # Thresholds Tab
    thresholds_frame = ttk.Frame(notebook, padding="10")
    notebook.add(thresholds_frame, text="Performance Thresholds")
    
    # Store entry widgets for later access
    weight_entries = {}
    threshold_entries = {}
    
    # Define descriptions for weights
    weight_descriptions = {
        "taxed_return": ("Taxed Return", "How much weight to give to after-tax returns. Higher = prioritize returns more."),
        "better_off": ("Better Off", "How much weight to give to outperforming buy-and-hold. Higher = prioritize beating the market more."),
        "win_rate": ("Win Rate", "How much weight to give to percentage of winning trades. Higher = prioritize consistency more."),
        "max_drawdown": ("Max Drawdown", "How much weight to give to avoiding large losses. Higher = penalize big drawdowns more."),
        "trade_count": ("Trade Count", "How much weight to give to having an optimal number of trades. Higher = prefer moderate trading frequency."),
        "hold_time": ("Hold Time", "How much weight to give to optimal holding periods. Higher = prefer certain holding durations."),
        "taxed_return_stability": ("Taxed Return Stability", "How much weight to give to consistent returns when parameters change slightly. Higher = prefer robust strategies."),
        "better_off_stability": ("Better Off Stability", "How much weight to give to consistent outperformance. Higher = prefer reliable edge."),
        "win_rate_stability": ("Win Rate Stability", "How much weight to give to consistent win rates. Higher = prefer stable performance.")
    }
    
    # Populate weights tab
    header_frame = ttk.Frame(weights_frame)
    header_frame.pack(fill="x", pady=(0, 10))
    ttk.Label(header_frame, text="Metric Weights", font=("Arial", 14, "bold")).pack()
    ttk.Label(header_frame, text="Adjust how much each metric contributes to the final score (0.0 to 1.0)", 
              font=("Arial", 9), foreground="gray").pack(pady=2)
    ttk.Label(header_frame, text="Tip: Weights should sum to approximately 1.0 for best results", 
              font=("Arial", 8, "italic"), foreground="blue").pack()
    
    # Create scrollable frame for weights
    weights_canvas = tk.Canvas(weights_frame)
    weights_scrollbar = ttk.Scrollbar(weights_frame, orient="vertical", command=weights_canvas.yview)
    weights_container = ttk.Frame(weights_canvas)
    
    weights_container.bind(
        "<Configure>",
        lambda e: weights_canvas.configure(scrollregion=weights_canvas.bbox("all"))
    )
    
    weights_canvas.create_window((0, 0), window=weights_container, anchor="nw")
    weights_canvas.configure(yscrollcommand=weights_scrollbar.set)
    
    weights_canvas.pack(side="left", fill="both", expand=True)
    weights_scrollbar.pack(side="right", fill="y")
    
    row = 0
    for metric, default_weight in scoring_config["weights"].items():
        frame = ttk.LabelFrame(weights_container, padding="8")
        frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        weights_container.columnconfigure(0, weight=1)
        
        # Metric name and description
        name, desc = weight_descriptions.get(metric, (metric.replace("_", " ").title(), ""))
        name_label = ttk.Label(frame, text=name, font=("Arial", 10, "bold"))
        name_label.grid(row=0, column=0, sticky="w", padx=5)
        
        desc_label = ttk.Label(frame, text=desc, font=("Arial", 8), foreground="gray", wraplength=600)
        desc_label.grid(row=1, column=0, sticky="w", padx=5, pady=(2, 5))
        
        # Value entry
        value_frame = ttk.Frame(frame)
        value_frame.grid(row=0, column=1, rowspan=2, sticky="e", padx=10)
        
        ttk.Label(value_frame, text="Weight:", font=("Arial", 9)).pack(side="left", padx=5)
        var = tk.DoubleVar(value=default_weight)
        entry = ttk.Entry(value_frame, textvariable=var, width=8)
        entry.pack(side="left", padx=5)
        
        # Show percentage
        percent_label = ttk.Label(value_frame, text=f"({default_weight*100:.1f}%)", 
                                  font=("Arial", 8), foreground="gray")
        percent_label.pack(side="left", padx=2)
        
        def update_percent(var=var, label=percent_label):
            try:
                val = float(var.get())
                label.config(text=f"({val*100:.1f}%)")
            except:
                pass
        
        var.trace("w", lambda *args: update_percent())
        weight_entries[metric] = var
        
        row += 1
    
    # Weight sum display
    weight_sum_frame = ttk.Frame(weights_frame)
    weight_sum_frame.pack(fill="x", pady=5)
    weight_sum_label = ttk.Label(weight_sum_frame, text="Total Weight: 1.00 (100%)", 
                                 font=("Arial", 10, "bold"))
    weight_sum_label.pack()
    
    def update_weight_sum():
        total = sum(var.get() for var in weight_entries.values())
        color = "green" if 0.95 <= total <= 1.05 else "orange" if 0.8 <= total <= 1.2 else "red"
        weight_sum_label.config(text=f"Total Weight: {total:.2f} ({total*100:.1f}%)", 
                               foreground=color)
    
    for var in weight_entries.values():
        var.trace("w", lambda *args: update_weight_sum())
    update_weight_sum()
    
    # Define descriptions for thresholds
    threshold_descriptions = {
        "taxed_return_excellent": ("Taxed Return - Excellent Threshold", 
                                   "Set this to define what you consider an 'excellent' return.\n\n"
                                   "If you set this to 0.5 (50%):\n"
                                   "  • Backtests with 50% return get full credit for this metric\n"
                                   "  • Backtests with 25% return get 50% credit (25% ÷ 50%)\n"
                                   "  • Backtests with 10% return get 20% credit (10% ÷ 50%)\n\n"
                                   "Think of it as: 'What return percentage deserves a perfect score?'"),
        "better_off_excellent": ("Better Off - Excellent Threshold", 
                                "Set this to define how much better than buy-and-hold is 'excellent'.\n\n"
                                "If you set this to 0.3 (30% better):\n"
                                "  • Backtests that beat buy-and-hold by 30% get full credit\n"
                                "  • Backtests that beat by 15% get 50% credit (15% ÷ 30%)\n"
                                "  • Backtests that beat by 5% get 17% credit (5% ÷ 30%)\n\n"
                                "Think of it as: 'How much outperformance deserves a perfect score?'"),
        "win_rate_excellent": ("Win Rate - Excellent Threshold", 
                              "Set this to define what win rate you consider 'excellent'.\n\n"
                              "If you set this to 0.6 (60% win rate):\n"
                              "  • Backtests with 60% win rate get full credit\n"
                              "  • Backtests with 30% win rate get 50% credit (30% ÷ 60%)\n"
                              "  • Backtests with 45% win rate get 75% credit (45% ÷ 60%)\n\n"
                              "Think of it as: 'What win rate percentage deserves a perfect score?'"),
        "max_drawdown_bad": ("Max Drawdown - Bad Threshold", 
                            "Set this to define what drawdown you consider 'bad' (worth zero points).\n\n"
                            "If you set this to 0.5 (50% drawdown):\n"
                            "  • Backtests with 50% drawdown get zero credit\n"
                            "  • Backtests with 25% drawdown get 50% credit (1 - 25% ÷ 50%)\n"
                            "  • Backtests with 10% drawdown get 80% credit (1 - 10% ÷ 50%)\n\n"
                            "Think of it as: 'What drawdown percentage is so bad it deserves zero points?'"),
        "trade_count_min": ("Trade Count - Minimum", 
                           "Minimum number of trades required. Backtests with fewer trades than this get zero points for trade count."),
        "trade_count_max": ("Trade Count - Maximum", 
                           "Maximum number of trades allowed. Backtests with more trades than this get reduced points."),
        "trade_count_optimal": ("Trade Count - Optimal", 
                               "The ideal number of trades. Backtests closest to this number get full points. "
                               "Points decrease as you move away from this number (toward min or max)."),
        "hold_time_min": ("Hold Time - Minimum (days)", 
                         "Minimum acceptable holding period in days. Backtests with average hold times below this get zero points."),
        "hold_time_max": ("Hold Time - Maximum (days)", 
                         "Maximum acceptable holding period in days. Backtests with average hold times above this get zero points."),
        "hold_time_optimal": ("Hold Time - Optimal (days)", 
                             "The ideal average holding period in days. Backtests closest to this get full points. "
                             "Points decrease as you move away from this number (toward min or max)."),
        "stability_threshold": ("Stability - Bad Threshold", 
                               "Set this to define when parameter stability is considered 'bad'.\n\n"
                               "If you set this to 0.1 (10% standard deviation):\n"
                               "  • Backtests with 10% std dev get zero stability credit\n"
                               "  • Backtests with 5% std dev get 50% credit (1 - 5% ÷ 10%)\n"
                               "  • Backtests with 2% std dev get 80% credit (1 - 2% ÷ 10%)\n\n"
                               "Lower standard deviation = more stable = better score. "
                               "Think of it as: 'What standard deviation is so high it deserves zero points?'")
    }
    
    # Populate thresholds tab
    header_frame2 = ttk.Frame(thresholds_frame)
    header_frame2.pack(fill="x", pady=(0, 10))
    ttk.Label(header_frame2, text="Performance Thresholds", font=("Arial", 14, "bold")).pack()
    ttk.Label(header_frame2, text="Define what 'excellent' or 'bad' performance means for each metric", 
              font=("Arial", 9), foreground="gray").pack(pady=2)
    ttk.Label(header_frame2, text="Values are typically decimals (0.5 = 50%) or counts (days, trades)", 
              font=("Arial", 8, "italic"), foreground="blue").pack()
    
    # Create scrollable frame for thresholds
    thresholds_canvas = tk.Canvas(thresholds_frame)
    thresholds_scrollbar = ttk.Scrollbar(thresholds_frame, orient="vertical", command=thresholds_canvas.yview)
    thresholds_container = ttk.Frame(thresholds_canvas)
    
    thresholds_container.bind(
        "<Configure>",
        lambda e: thresholds_canvas.configure(scrollregion=thresholds_canvas.bbox("all"))
    )
    
    thresholds_canvas.create_window((0, 0), window=thresholds_container, anchor="nw")
    thresholds_canvas.configure(yscrollcommand=thresholds_scrollbar.set)
    
    thresholds_canvas.pack(side="left", fill="both", expand=True)
    thresholds_scrollbar.pack(side="right", fill="y")
    
    row = 0
    for threshold, default_value in scoring_config["thresholds"].items():
        frame = ttk.LabelFrame(thresholds_container, padding="8")
        frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        thresholds_container.columnconfigure(0, weight=1)
        
        # Threshold name and description
        name, desc = threshold_descriptions.get(threshold, (threshold.replace("_", " ").title(), ""))
        name_label = ttk.Label(frame, text=name, font=("Arial", 10, "bold"))
        name_label.grid(row=0, column=0, sticky="w", padx=5)
        
        desc_label = ttk.Label(frame, text=desc, font=("Arial", 8), foreground="gray", wraplength=700)
        desc_label.grid(row=1, column=0, sticky="w", padx=5, pady=(2, 5))
        
        # Value entry with example
        value_frame = ttk.Frame(frame)
        value_frame.grid(row=0, column=1, rowspan=2, sticky="e", padx=10)
        
        ttk.Label(value_frame, text="Value:", font=("Arial", 9)).pack(side="left", padx=5)
        var = tk.DoubleVar(value=default_value)
        entry = ttk.Entry(value_frame, textvariable=var, width=10)
        entry.pack(side="left", padx=5)
        
        # Show example interpretation
        if "excellent" in threshold or "bad" in threshold or threshold == "stability_threshold":
            # Show as percentage
            example_text = f"({default_value*100:.0f}%)" if default_value < 1 else f"({default_value})"
        elif "optimal" in threshold or "min" in threshold or "max" in threshold:
            # Show as count
            example_text = f"({int(default_value)} {('days' if 'time' in threshold else 'trades')})"
        else:
            example_text = ""
        
        if example_text:
            example_label = ttk.Label(value_frame, text=example_text, 
                                     font=("Arial", 8), foreground="gray")
            example_label.pack(side="left", padx=2)
        
        threshold_entries[threshold] = var
        
        row += 1
    
    # Buttons frame
    buttons_frame = ttk.Frame(config_window)
    buttons_frame.pack(fill="x", padx=10, pady=10)
    
    def save_config():
        global scoring_config
        # Validate and update weights
        new_weights = {}
        weight_errors = []
        for metric, var in weight_entries.items():
            try:
                val = float(var.get())
                if val < 0:
                    weight_errors.append(f"{metric}: Weight cannot be negative")
                new_weights[metric] = max(0.0, val)
            except ValueError:
                weight_errors.append(f"{metric}: Invalid number")
                new_weights[metric] = scoring_config["weights"][metric]
        
        # Validate and update thresholds
        new_thresholds = {}
        threshold_errors = []
        for threshold, var in threshold_entries.items():
            try:
                val = float(var.get())
                if val < 0:
                    threshold_errors.append(f"{threshold}: Value cannot be negative")
                new_thresholds[threshold] = max(0.0, val)
            except ValueError:
                threshold_errors.append(f"{threshold}: Invalid number")
                new_thresholds[threshold] = scoring_config["thresholds"][threshold]
        
        # Check for logical errors
        if "trade_count_min" in new_thresholds and "trade_count_max" in new_thresholds:
            if new_thresholds["trade_count_min"] > new_thresholds["trade_count_max"]:
                threshold_errors.append("Trade Count Min cannot be greater than Max")
        
        if "hold_time_min" in new_thresholds and "hold_time_max" in new_thresholds:
            if new_thresholds["hold_time_min"] > new_thresholds["hold_time_max"]:
                threshold_errors.append("Hold Time Min cannot be greater than Max")
        
        if weight_errors or threshold_errors:
            error_msg = "Please fix the following errors:\n\n"
            if weight_errors:
                error_msg += "Weights:\n" + "\n".join(f"  • {e}" for e in weight_errors) + "\n\n"
            if threshold_errors:
                error_msg += "Thresholds:\n" + "\n".join(f"  • {e}" for e in threshold_errors)
            messagebox.showerror("Validation Error", error_msg)
            return
        
        scoring_config = {
            "weights": new_weights,
            "thresholds": new_thresholds
        }
        
        messagebox.showinfo("Success", "Scoring configuration saved!\n\nChanges will apply to:\n• New backtests\n• Rescored cached backtests")
        config_window.destroy()
    
    def reset_to_defaults():
        global scoring_config
        scoring_config = scoring.get_default_scoring_config()
        # Update UI
        for metric, var in weight_entries.items():
            var.set(scoring_config["weights"][metric])
        for threshold, var in threshold_entries.items():
            var.set(scoring_config["thresholds"][threshold])
        update_weight_sum()
        messagebox.showinfo("Reset", "Reset to default values")
    
    def show_help():
        help_text = """SCORING CONFIGURATION HELP

METRIC WEIGHTS:
• Weights determine how much each metric contributes to the final score (0-10)
• Higher weight = that metric has more influence
• Weights should sum to approximately 1.0 for best results
• Example: If Taxed Return weight is 0.25, it contributes 25% to the score

PERFORMANCE THRESHOLDS:
• Thresholds define what "excellent" or "bad" performance means
• Decimal values (0.0-1.0) represent percentages
  - 0.5 = 50%
  - 0.3 = 30%
  - 1.0 = 100%
• Count values represent actual numbers (trades, days)

EXAMPLES:
• Taxed Return Excellent = 0.5 means:
  - A 50% return gets full points (10/10 for that metric)
  - A 25% return gets half points (5/10 for that metric)
  - A 10% return gets 2/10 points

• Win Rate Excellent = 0.6 means:
  - A 60% win rate gets full points
  - A 30% win rate gets half points

• Max Drawdown Bad = 0.5 means:
  - A 50% drawdown gets zero points
  - A 25% drawdown gets half points (less penalty)

SCORING:
• Each metric is scored 0-1 based on thresholds
• Scores are multiplied by weights
• Final score is normalized to 0-10 scale
• Higher score = better overall strategy"""
        
        help_window = tk.Toplevel(config_window)
        help_window.title("Scoring Configuration Help")
        help_window.geometry("600x700")
        
        text_widget = tk.Text(help_window, wrap="word", padx=10, pady=10, font=("Arial", 9))
        text_widget.pack(fill="both", expand=True)
        text_widget.insert("1.0", help_text)
        text_widget.config(state="disabled")
        
        ttk.Button(help_window, text="Close", command=help_window.destroy).pack(pady=5)
    
    ttk.Button(buttons_frame, text="Help", command=show_help).pack(side="left", padx=5)
    ttk.Button(buttons_frame, text="Reset to Defaults", command=reset_to_defaults).pack(side="left", padx=5)
    ttk.Button(buttons_frame, text="Cancel", command=config_window.destroy).pack(side="left", padx=5)
    ttk.Button(buttons_frame, text="Save", command=save_config).pack(side="right", padx=5)

def view_cached_backtests():
    """Open a comprehensive window to view cached backtest results with trades."""
    from pathlib import Path
    import pickle
    import cache_manager
    
    cache_dir = Path("backtest_cache")
    
    # Get root cache files and batch folders
    root_cache_files = list(cache_dir.glob("*.pkl")) if cache_dir.exists() else []
    batch_folders = [d for d in cache_dir.iterdir() if d.is_dir() and not d.name.startswith('.')] if cache_dir.exists() else []
    
    if not root_cache_files and not batch_folders:
        messagebox.showinfo("No Cache", "No cached backtest results found.")
        return
    
    # Create cache viewer window - larger for comprehensive view
    cache_window = tk.Toplevel(root)
    cache_window.title("View Cached Backtests - Comprehensive Results")
    cache_window.geometry("1800x900")
    
    # Top frame for source selection
    source_frame = ttk.Frame(cache_window, padding="10")
    source_frame.pack(fill="x", padx=10, pady=5)
    
    ttk.Label(source_frame, text="Source:", font=("Arial", 12)).pack(side="left", padx=5)
    source_var = tk.StringVar(value="Root Cache Files" if root_cache_files else "Batch Folders")
    source_radio1 = ttk.Radiobutton(source_frame, text="Root Cache Files", variable=source_var, value="Root Cache Files")
    source_radio1.pack(side="left", padx=10)
    source_radio2 = ttk.Radiobutton(source_frame, text="Batch Folders", variable=source_var, value="Batch Folders")
    source_radio2.pack(side="left", padx=10)
    
    # Frame for batch folder selection (hidden initially)
    batch_folder_frame = ttk.Frame(cache_window, padding="5")
    batch_folder_frame.pack(fill="x", padx=10, pady=5)
    
    ttk.Label(batch_folder_frame, text="Select Batch Folder:", font=("Arial", 11)).pack(side="left", padx=5)
    batch_folder_var = tk.StringVar()
    batch_folder_names = [f"{f.name} ({len(list(f.glob('*.pkl')))} files)" for f in batch_folders]
    batch_folder_combobox = ttk.Combobox(
        batch_folder_frame,
        textvariable=batch_folder_var,
        values=batch_folder_names,
        state="readonly",
        width=50
    )
    batch_folder_combobox.pack(side="left", padx=5)
    if batch_folder_names:
        batch_folder_combobox.current(0)
    
    # Top frame for cache file selection
    top_frame = ttk.Frame(cache_window, padding="10")
    top_frame.pack(fill="x", padx=10, pady=10)
    
    ttk.Label(top_frame, text="Select Cache File:", font=("Arial", 12)).pack(side="left", padx=5)
    
    cache_file_var = tk.StringVar()
    cache_file_names = [f"{f.name} ({f.stat().st_size / 1024:.1f} KB)" for f in root_cache_files] if root_cache_files else []
    cache_combobox = ttk.Combobox(
        top_frame, 
        textvariable=cache_file_var, 
        values=cache_file_names, 
        state="readonly", 
        width=50
    )
    cache_combobox.pack(side="left", padx=5)
    if cache_file_names:
        cache_combobox.current(0)  # Select first file by default
    
    # Store current cache files list and paths
    current_cache_files = root_cache_files.copy()
    current_cache_paths = {f"{f.name} ({f.stat().st_size / 1024:.1f} KB)": f for f in root_cache_files}
    
    def update_source_display():
        """Update the display based on selected source."""
        if source_var.get() == "Root Cache Files":
            batch_folder_frame.pack_forget()
            top_frame.pack(fill="x", padx=10, pady=10, after=source_frame)
            
            # Update cache files list
            current_cache_files.clear()
            current_cache_files.extend(root_cache_files)
            current_cache_paths.clear()
            for f in root_cache_files:
                display_name = f"{f.name} ({f.stat().st_size / 1024:.1f} KB)"
                current_cache_paths[display_name] = f
            
            cache_file_names = [f"{f.name} ({f.stat().st_size / 1024:.1f} KB)" for f in root_cache_files]
            cache_combobox['values'] = cache_file_names
            if cache_file_names:
                cache_combobox.current(0)
            else:
                cache_combobox.set("")
        else:  # Batch Folders
            top_frame.pack_forget()
            batch_folder_frame.pack(fill="x", padx=10, pady=5, after=source_frame)
            
            # Update batch folder list
            batch_folders = [d for d in cache_dir.iterdir() if d.is_dir() and not d.name.startswith('.')] if cache_dir.exists() else []
            batch_folder_names = [f"{f.name} ({len(list(f.glob('*.pkl')))} files)" for f in batch_folders]
            batch_folder_combobox['values'] = batch_folder_names
            if batch_folder_names:
                batch_folder_combobox.current(0)
                update_cache_files_from_batch()
            else:
                batch_folder_combobox.set("")
                cache_file_names = []
                cache_combobox['values'] = cache_file_names
                cache_combobox.set("")
    
    def update_cache_files_from_batch():
        """Update cache files list based on selected batch folder."""
        selection = batch_folder_combobox.current()
        if selection < 0:
            return
        
        batch_folders = [d for d in cache_dir.iterdir() if d.is_dir() and not d.name.startswith('.')] if cache_dir.exists() else []
        if selection >= len(batch_folders):
            return
        
        selected_folder = batch_folders[selection]
        batch_cache_files = list(selected_folder.glob("*.pkl"))
        
        # Update cache files list
        current_cache_files.clear()
        current_cache_files.extend(batch_cache_files)
        current_cache_paths.clear()
        for f in batch_cache_files:
            display_name = f"{f.name} ({f.stat().st_size / 1024:.1f} KB)"
            current_cache_paths[display_name] = f
        
        cache_file_names = [f"{f.name} ({f.stat().st_size / 1024:.1f} KB)" for f in batch_cache_files]
        cache_combobox['values'] = cache_file_names
        
        # Show the cache file selection frame
        top_frame.pack(fill="x", padx=10, pady=10, after=batch_folder_frame)
        
        if cache_file_names:
            cache_combobox.current(0)
            load_cache_file()
        else:
            cache_combobox.set("")
    
    # Bind source change
    source_radio1.config(command=update_source_display)
    source_radio2.config(command=update_source_display)
    batch_folder_combobox.bind("<<ComboboxSelected>>", lambda e: update_cache_files_from_batch())
    
    # Initial display
    if not root_cache_files and batch_folders:
        source_var.set("Batch Folders")
    update_source_display()
    
    # Info frame for cache metadata
    info_frame = ttk.LabelFrame(cache_window, text="Cache Information", padding="10")
    info_frame.pack(fill="x", padx=10, pady=5)
    
    info_text = tk.Text(info_frame, height=6, wrap="word", state="disabled")
    info_text.pack(fill="x", padx=5, pady=5)
    
    # Create paned window for split view (combinations on left, trades on right)
    paned = ttk.PanedWindow(cache_window, orient="horizontal")
    paned.pack(fill="both", expand=True, padx=10, pady=5)
    
    # Left pane: Combinations table
    left_pane = ttk.Frame(paned)
    paned.add(left_pane, weight=1)
    
    # Tree frame for combinations
    tree_frame_cache = ttk.Frame(left_pane)
    tree_frame_cache.pack(fill="both", expand=True)
    
    tree_scroll_y_cache = ttk.Scrollbar(tree_frame_cache, orient="vertical")
    tree_scroll_y_cache.pack(side="right", fill="y")
    tree_scroll_x_cache = ttk.Scrollbar(tree_frame_cache, orient="horizontal")
    tree_scroll_x_cache.pack(side="bottom", fill="x")
    
    tree_cache = ttk.Treeview(
        tree_frame_cache,
        yscrollcommand=tree_scroll_y_cache.set,
        xscrollcommand=tree_scroll_x_cache.set,
        columns=("Backtest_Score", "SMA_A", "SMA_B", "Taxed_Return", "Better_Off", "Win_Rate", "Trade_Count", 
                "Winning_Trades", "Losing_Trades", "Max_Drawdown", "Avg_Hold_Time", "Avg_Trade_Return",
                "Return_Std", "End_Liquidity", "Under1Y_PL", "Over1Y_PL", "Win_Pct_Last4"),
        show="headings"
    )
    tree_scroll_y_cache.config(command=tree_cache.yview)
    tree_scroll_x_cache.config(command=tree_cache.xview)
    
    # Configure columns
    tree_cache.heading("Backtest_Score", text="Score")
    tree_cache.heading("SMA_A", text="SMA A")
    tree_cache.heading("SMA_B", text="SMA B")
    tree_cache.heading("Taxed_Return", text="Taxed Return")
    tree_cache.heading("Better_Off", text="Better Off")
    tree_cache.heading("Win_Rate", text="Win Rate")
    tree_cache.heading("Trade_Count", text="Trades")
    tree_cache.heading("Winning_Trades", text="Wins")
    tree_cache.heading("Losing_Trades", text="Losses")
    tree_cache.heading("Max_Drawdown", text="Max Drawdown")
    tree_cache.heading("Avg_Hold_Time", text="Avg Hold (days)")
    tree_cache.heading("Avg_Trade_Return", text="Avg Trade Return")
    tree_cache.heading("Return_Std", text="Return Std")
    tree_cache.heading("End_Liquidity", text="End Liquidity")
    tree_cache.heading("Under1Y_PL", text="Under 1Y P/L")
    tree_cache.heading("Over1Y_PL", text="Over 1Y P/L")
    tree_cache.heading("Win_Pct_Last4", text="Win % Last 4")
    
    # Set column widths
    tree_cache.column("Backtest_Score", width=120, anchor="center")
    tree_cache.column("SMA_A", width=60, anchor="center")
    tree_cache.column("SMA_B", width=60, anchor="center")
    tree_cache.column("Taxed_Return", width=100, anchor="center")
    tree_cache.column("Better_Off", width=100, anchor="center")
    tree_cache.column("Win_Rate", width=80, anchor="center")
    tree_cache.column("Trade_Count", width=70, anchor="center")
    tree_cache.column("Winning_Trades", width=70, anchor="center")
    tree_cache.column("Losing_Trades", width=70, anchor="center")
    tree_cache.column("Max_Drawdown", width=100, anchor="center")
    tree_cache.column("Avg_Hold_Time", width=100, anchor="center")
    tree_cache.column("Avg_Trade_Return", width=120, anchor="center")
    tree_cache.column("Return_Std", width=90, anchor="center")
    tree_cache.column("End_Liquidity", width=110, anchor="center")
    tree_cache.column("Under1Y_PL", width=100, anchor="center")
    tree_cache.column("Over1Y_PL", width=100, anchor="center")
    tree_cache.column("Win_Pct_Last4", width=100, anchor="center")
    
    tree_cache.pack(fill="both", expand=True)
    
    # Right pane: Trades viewer
    right_pane = ttk.LabelFrame(paned, text="Trades for Selected Combination", padding="5")
    paned.add(right_pane, weight=1)
    
    # Move tree to left pane
    tree_frame_cache = ttk.Frame(left_pane)
    tree_frame_cache.pack(fill="both", expand=True)
    
    # Bottom frame for actions (in left pane)
    bottom_frame = ttk.Frame(left_pane, padding="10")
    bottom_frame.pack(fill="x", padx=5, pady=5)
    
    # Trades tree in right pane
    trades_frame = ttk.Frame(right_pane)
    trades_frame.pack(fill="both", expand=True, padx=5, pady=5)
    
    # Explanation label for walk-forward trades
    trades_explanation = ttk.Label(right_pane, text="", foreground="blue", font=("Arial", 9), wraplength=600)
    trades_explanation.pack(pady=5, padx=5)
    
    trades_scroll_y = ttk.Scrollbar(trades_frame, orient="vertical")
    trades_scroll_y.pack(side="right", fill="y")
    trades_scroll_x = ttk.Scrollbar(trades_frame, orient="horizontal")
    trades_scroll_x.pack(side="bottom", fill="x")
    
    trades_tree = ttk.Treeview(
        trades_frame,
        yscrollcommand=trades_scroll_y.set,
        xscrollcommand=trades_scroll_x.set,
        show="headings"
    )
    trades_tree.pack(fill="both", expand=True)
    trades_scroll_y.config(command=trades_tree.yview)
    trades_scroll_x.config(command=trades_tree.xview)
    
    # Status label for trades
    trades_status = ttk.Label(right_pane, text="Select a combination to view trades", foreground="gray")
    trades_status.pack(pady=5)
    
    # Store current cache data for trade viewing
    current_cache_data = None
    current_sorted_combos = []
    
    def show_trades_for_combination(event=None):
        """Show trades for the selected combination."""
        selection = tree_cache.selection()
        if not selection:
            trades_status.config(text="Select a combination to view trades")
            return
        
        item = selection[0]
        # Find the index in sorted_combos
        item_index = tree_cache.index(item)
        if item_index >= len(current_sorted_combos):
            return
        
        # Unpack based on tuple structure: (combo, backtest_score, walk_forward_score, combined_score)
        score_data = current_sorted_combos[item_index]
        if len(score_data) == 4:
            combo, backtest_score, walk_forward_score, combined_score = score_data
        else:
            # Fallback for old format (combo, score)
            combo, backtest_score = score_data[0], score_data[1] if len(score_data) > 1 else 0.0
        
        # Clear trades tree
        for item in trades_tree.get_children():
            trades_tree.delete(item)
        trades_tree["columns"] = ()
        
        # Check if trades are stored for this combination
        # For now, only best combination has trades stored
        if current_cache_data:
            best_idx = current_cache_data.get('best_combination_idx', 0)
            all_combinations = current_cache_data.get('all_combinations', [])
            is_best = combo == all_combinations[best_idx] if best_idx < len(all_combinations) else False
            
            # Check if this is walk-forward - display trades by segment
            if current_cache_data.get("walk_forward_mode", False):
                walk_forward_segment_trades = current_cache_data.get('walk_forward_segment_trades', [])
                if not walk_forward_segment_trades:
                    trades_explanation.config(text="")
                    # Check if this might be an old cache file
                    cached_at = current_cache_data.get('cached_at', '')
                    if cached_at:
                        trades_status.config(text=f"Walk-Forward Analysis: No segment trades available. This cache file may have been created before segment trades were stored. (Cached: {cached_at})")
                    else:
                        trades_status.config(text="Walk-Forward Analysis: No segment trades available. This may be an older cache file.")
                    return
                
                # Show explanation
                explanation_text = (
                    "WALK-FORWARD ANALYSIS EXPLANATION:\n"
                    "• Each segment has a TRAINING period (to find best parameters) and a TEST period (to test those parameters)\n"
                    "• ALL trades shown below are from the WALK-FORWARD (TEST) periods only\n"
                    "• Training period trades are NOT shown (they're only used for optimization)\n"
                    "• The 'Test Period' column shows the date range when these trades occurred"
                )
                trades_explanation.config(text=explanation_text)
                
                # Display walk-forward trades separated by segment
                all_trades = []
                for seg_data in walk_forward_segment_trades:
                    segment_num = seg_data.get('segment', 0)
                    seg_trades = seg_data.get('trades', [])
                    test_start = seg_data.get('test_start', '')
                    test_end = seg_data.get('test_end', '')
                    train_start = seg_data.get('train_start', '')
                    train_end = seg_data.get('train_end', '')
                    
                    # Format dates
                    if isinstance(test_start, datetime):
                        test_period_str = f"{test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}"
                    else:
                        test_period_str = f"{test_start} to {test_end}"
                    
                    if isinstance(train_start, datetime):
                        train_period_str = f"{train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}"
                    else:
                        train_period_str = f"{train_start} to {train_end}"
                    
                    # Add trades with segment info
                    if seg_trades:
                        for trade in seg_trades:
                            # Format dates if they're datetime objects
                            if isinstance(trade.get('Date'), datetime):
                                trade_date = trade['Date'].strftime('%Y-%m-%d')
                            else:
                                trade_date = str(trade.get('Date', ''))
                            
                            if isinstance(trade.get('BuyDate'), datetime):
                                buy_date = trade['BuyDate'].strftime('%Y-%m-%d')
                            else:
                                buy_date = str(trade.get('BuyDate', ''))
                            
                            if isinstance(trade.get('SellDate'), datetime):
                                sell_date = trade['SellDate'].strftime('%Y-%m-%d')
                            else:
                                sell_date = str(trade.get('SellDate', ''))
                            
                            all_trades.append({
                                'Segment': f"Segment {segment_num}",
                                'Training Period': train_period_str,
                                'Walk-Forward Test Period': test_period_str,
                                'Trade Date': trade_date,
                                'BuyDate': buy_date,
                                'SellDate': sell_date,
                                'BuyPrice': f"{trade.get('BuyPrice', 0):.2f}",
                                'SellPrice': f"{trade.get('SellPrice', 0):.2f}",
                                'PreTaxReturn': f"{trade.get('PreTaxReturn', 0):.4f}",
                                'HoldTime': trade.get('HoldTime', 0),
                                'GainDollars': f"${trade.get('GainDollars', 0):.2f}",
                                'SMA_A': trade.get('SMA_A', ''),
                                'SMA_B': trade.get('SMA_B', '')
                            })
                
                if all_trades:
                    # Set up columns
                    columns = ['Segment', 'Training Period', 'Walk-Forward Test Period', 'Trade Date', 'BuyDate', 'SellDate', 
                              'BuyPrice', 'SellPrice', 'PreTaxReturn', 'HoldTime', 'GainDollars', 'SMA_A', 'SMA_B']
                    trades_tree["columns"] = columns
                    for col in columns:
                        trades_tree.heading(col, text=col)
                        trades_tree.column(col, anchor="center", width=100)
                    
                    # Insert trades
                    for trade in all_trades:
                        values = [str(trade.get(col, '')) for col in columns]
                        trades_tree.insert("", "end", values=values)
                    
                    total_trades = len(all_trades)
                    segments_count = len(walk_forward_segment_trades)
                    trades_status.config(text=f"✓ Showing {total_trades} WALK-FORWARD (TEST) trades across {segments_count} segments")
                else:
                    trades_explanation.config(text="")
                    trades_status.config(text="Walk-Forward Analysis: No trades found in segments.")
                return
            else:
                # Not walk-forward, clear explanation
                trades_explanation.config(text="")
            
            # Try to get trades from cache (if stored)
            trades = None
            # First check if trades are stored in the combination itself
            if 'trades' in combo and combo.get('trades'):
                trades = combo.get('trades', [])
                # Convert Date strings back to datetime objects if needed
                if trades and isinstance(trades[0].get('Date', ''), str):
                    for trade in trades:
                        if 'Date' in trade and isinstance(trade['Date'], str):
                            try:
                                trade['Date'] = datetime.strptime(trade['Date'], '%Y-%m-%d')
                            except:
                                pass
            # Fallback to besttrades for backward compatibility
            elif 'besttrades' in current_cache_data and is_best:
                trades = current_cache_data.get('besttrades', [])
            
            if trades:
                # Display trades
                if isinstance(trades, list) and len(trades) > 0:
                    columns = list(trades[0].keys())
                    trades_tree["columns"] = columns
                    for col in columns:
                        trades_tree.heading(col, text=col)
                        trades_tree.column(col, anchor="center", width=100)
                    
                    for trade in trades:
                        values = [str(trade.get(col, '')) for col in columns]
                        trades_tree.insert("", "end", values=values)
                    
                    trades_status.config(text=f"Showing {len(trades)} trades for SMA {combo.get('sma_a', 'N/A')}, {combo.get('sma_b', 'N/A')}")
                else:
                    trades_status.config(text="No trades available for this combination")
            else:
                # For older cache files that don't have trades stored per combination
                trades_status.config(text=f"Trades not available for this combination. (SMA: {combo.get('sma_a', 'N/A')}, {combo.get('sma_b', 'N/A')}) - This may be an older cache file.")
    
    def load_cache_file(event=None):
        """Load and display the selected cache file."""
        selection = cache_combobox.current()
        if selection < 0:
            return
        
        if selection >= len(current_cache_files):
            return
        
        cache_file = current_cache_files[selection]
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Update info text
            info_text.config(state="normal")
            info_text.delete(1.0, tk.END)
            info_text.insert(tk.END, f"Ticker: {cache_data.get('ticker', 'N/A')}\n")
            info_text.insert(tk.END, f"Start Date: {cache_data.get('start_date', 'N/A')}\n")
            info_text.insert(tk.END, f"End Date: {cache_data.get('end_date', 'N/A')}\n")
            info_text.insert(tk.END, f"Compounding: {cache_data.get('compounding', 'N/A')}\n")
            info_text.insert(tk.END, f"Optimization Objective: {cache_data.get('optimization_objective', 'N/A')}\n")
            info_text.insert(tk.END, f"Start Amount: ${cache_data.get('start_amount', 0):,.2f}\n")
            info_text.insert(tk.END, f"No Algorithm Return: {cache_data.get('noalgoreturn', 0):.4%}\n")
            info_text.insert(tk.END, f"Cached At: {cache_data.get('cached_at', 'N/A')}\n")
            info_text.insert(tk.END, f"Total Combinations: {len(cache_data.get('all_combinations', []))}\n")
            if cache_data.get("walk_forward_mode", False):
                info_text.insert(tk.END, f"\n⭐ WALK-FORWARD ANALYSIS ⭐\n")
                info_text.insert(tk.END, f"Segments: {cache_data.get('segments', 0)}\n")
                info_text.insert(tk.END, f"Training Score: {cache_data.get('training_score', 0):.2f}/10.0\n")
                info_text.insert(tk.END, f"Walk-Forward Score: {cache_data.get('walk_forward_score', 0):.2f}/10.0\n")
                info_text.insert(tk.END, f"Combined Score: {cache_data.get('combined_score', 0):.2f}/10.0 (40% training + 60% walk-forward)\n")
            info_text.config(state="disabled")
            
            # Store cache data for trade viewing
            nonlocal current_cache_data, current_sorted_combos
            current_cache_data = cache_data
            
            # Clear and populate tree
            for item in tree_cache.get_children():
                tree_cache.delete(item)
            
            all_combinations = cache_data.get('all_combinations', [])
            best_idx = cache_data.get('best_combination_idx', 0)
            
            # Calculate scores for all combinations using scoring algorithm
            combo_scores = []
            is_walk_forward = cache_data.get("walk_forward_mode", False)
            
            # Debug: Print walk-forward mode status
            print(f"DEBUG: Loading cache file - is_walk_forward: {is_walk_forward}")
            print(f"DEBUG: Number of combinations: {len(all_combinations)}")
            if all_combinations:
                print(f"DEBUG: First combo keys: {list(all_combinations[0].keys())}")
            
            for combo_idx, combo in enumerate(all_combinations):
                # Build combo_result for scoring calculation
                combo_result = {
                    "outputresults1": {
                        "besttaxedreturn": combo.get("taxed_return", 0),
                        "betteroff": combo.get("better_off", 0),
                        "besttradecount": combo.get("trade_count", 0),
                        "noalgoreturn": cache_data.get("noalgoreturn", 0)
                    },
                    "outputresults2": {
                        "winningtradepct": combo.get("win_rate", 0),
                        "maxdrawdown(worst trade return pct)": combo.get("max_drawdown", 0),
                        "average_hold_time": combo.get("avg_hold_time", 0)
                    },
                    "param_stability": {
                        "taxed_return_std": 0,  # Individual combo doesn't have stability
                        "better_off_std": 0,
                        "win_rate_std": 0,
                        "taxed_return_max_min_diff": 0
                    }
                }
                
                if is_walk_forward:
                    # For walk-forward: scores are stored in each combo from walk_forward.py
                    # Get stored scores directly from combo (they're already calculated from trade time intervals)
                    # Check if keys exist in combo first, then get values
                    if 'training_score' in combo:
                        training_score = combo['training_score']
                    else:
                        training_score = cache_data.get('training_score')
                    
                    if 'walk_forward_score' in combo:
                        walk_forward_score = combo['walk_forward_score']
                    else:
                        walk_forward_score = cache_data.get('walk_forward_score')
                    
                    if 'combined_score' in combo:
                        combined_score = combo['combined_score']
                    else:
                        combined_score = cache_data.get('combined_score')
                    
                    # Debug: Print first combo to verify scores are being retrieved
                    if combo_idx == 0:  # First combo
                        print(f"DEBUG: Retrieving scores from combo (walk-forward mode):")
                        print(f"  Combo keys: {list(combo.keys())[:15]}...")  # First 15 keys
                        print(f"  Has training_score key: {'training_score' in combo}")
                        print(f"  Has walk_forward_score key: {'walk_forward_score' in combo}")
                        print(f"  Has combined_score key: {'combined_score' in combo}")
                        if 'training_score' in combo:
                            print(f"  training_score value: {combo['training_score']} (type: {type(combo['training_score'])})")
                        if 'walk_forward_score' in combo:
                            print(f"  walk_forward_score value: {combo['walk_forward_score']} (type: {type(combo['walk_forward_score'])})")
                        if 'combined_score' in combo:
                            print(f"  combined_score value: {combo['combined_score']} (type: {type(combo['combined_score'])})")
                        print(f"  Retrieved - training: {training_score}, walk_forward: {walk_forward_score}, combined: {combined_score}")
                    
                    # If still missing (None), recalculate using current scoring config
                    # Note: 0.0 is a valid score, so we only check for None
                    if training_score is None:
                        # Try to recalculate training score (but we don't have training period data here)
                        # So use 0.0 as placeholder
                        training_score = 0.0
                    
                    if walk_forward_score is None:
                        # Recalculate walk-forward score from test period metrics using current scoring config
                        walk_forward_result = {
                            "outputresults1": {
                                "besttaxedreturn": combo.get("taxed_return", 0),
                                "betteroff": combo.get("better_off", 0),
                                "besttradecount": combo.get("trade_count", 0),
                                "noalgoreturn": cache_data.get("noalgoreturn", 0)
                            },
                            "outputresults2": {
                                "winningtradepct": combo.get("win_rate", 0),
                                "maxdrawdown(worst trade return pct)": combo.get("max_drawdown", 0),
                                "average_hold_time": combo.get("avg_hold_time", 0)
                            },
                            "param_stability": {
                                "taxed_return_std": combo.get("taxed_return_std", 0) if 'taxed_return_std' in combo else 0,
                                "better_off_std": 0,
                                "win_rate_std": 0,
                                "taxed_return_max_min_diff": combo.get("taxed_return_max_min_diff", 0) if 'taxed_return_max_min_diff' in combo else 0
                            }
                        }
                        walk_forward_score = scoring.calculate_backtest_score(walk_forward_result, scoring_config)
                    
                    # Calculate combined score if missing (None only, 0.0 is valid)
                    if combined_score is None:
                        if training_score is not None and walk_forward_score is not None:
                            combined_score = (training_score * 0.4 + walk_forward_score * 0.6)
                        elif walk_forward_score is not None:
                            combined_score = walk_forward_score
                        else:
                            combined_score = 0.0
                    
                    # Ensure all scores are numbers, not None (convert None to 0.0)
                    # This ensures they display correctly in the table
                    try:
                        training_score = float(training_score) if training_score is not None else 0.0
                        walk_forward_score = float(walk_forward_score) if walk_forward_score is not None else 0.0
                        combined_score = float(combined_score) if combined_score is not None else 0.0
                    except (ValueError, TypeError) as e:
                        # Debug: print error if conversion fails
                        print(f"Error converting scores: {e}, training_score={training_score}, walk_forward_score={walk_forward_score}, combined_score={combined_score}")
                        training_score = 0.0
                        walk_forward_score = 0.0
                        combined_score = 0.0
                    
                    # Ensure scores are never None (should be numbers at this point)
                    if training_score is None:
                        training_score = 0.0
                    if walk_forward_score is None:
                        walk_forward_score = 0.0
                    if combined_score is None:
                        combined_score = 0.0
                    
                    # Debug: Print what we're appending
                    if combo_idx == 0:
                        print(f"DEBUG: Appending to combo_scores - training: {training_score}, walk_forward: {walk_forward_score}, combined: {combined_score}")
                    
                    combo_scores.append((combo, training_score, walk_forward_score, combined_score))
                else:
                    # For standard backtest: calculate backtest score using scoring algorithm
                    backtest_score = scoring.calculate_backtest_score(combo_result, scoring_config)
                    combo_scores.append((combo, backtest_score, None, None))
            
            # Sort by combined score (if walk-forward) or backtest score, then by taxed return
            if is_walk_forward:
                sorted_combos = sorted(combo_scores, key=lambda x: (x[3] if x[3] is not None else x[1], x[0].get('taxed_return', -999)), reverse=True)
            else:
                sorted_combos = sorted(combo_scores, key=lambda x: (x[1], x[0].get('taxed_return', -999)), reverse=True)
            current_sorted_combos = sorted_combos
            
            for i, score_data in enumerate(sorted_combos):
                # Unpack score data - format is (combo, backtest_score/training_score, walk_forward_score, combined_score)
                if len(score_data) == 4:
                    combo, backtest_score, walk_forward_score, combined_score = score_data
                else:
                    # Fallback for old format
                    combo = score_data[0]
                    backtest_score = score_data[1] if len(score_data) > 1 else 0.0
                    walk_forward_score = score_data[2] if len(score_data) > 2 else None
                    combined_score = score_data[3] if len(score_data) > 3 else None
                
                # Debug: Print first combo to verify scores are being unpacked correctly
                if i == 0 and is_walk_forward:
                    print(f"DEBUG: Unpacked scores - backtest: {backtest_score}, walk_forward: {walk_forward_score}, combined: {combined_score}")
                    print(f"DEBUG: Types - backtest: {type(backtest_score)}, walk_forward: {type(walk_forward_score)}, combined: {type(combined_score)}")
                
                # Format values for all columns
                win_pct_last4 = combo.get('win_pct_last_4', None)
                win_pct_last4_str = f"{win_pct_last4:.2%}" if win_pct_last4 is not None else "N/A"
                
                # Format score columns - ensure they're numbers before formatting
                # For walk-forward: backtest_score is actually training_score
                try:
                    if backtest_score is not None:
                        backtest_score_str = f"{float(backtest_score):.2f}"
                    else:
                        backtest_score_str = "N/A"
                except (ValueError, TypeError):
                    backtest_score_str = "N/A"
                
                values = (
                    backtest_score_str,
                    combo.get('sma_a', ''),
                    combo.get('sma_b', ''),
                    f"{combo.get('taxed_return', 0):.4%}",
                    f"{combo.get('better_off', 0):.4%}",
                    f"{combo.get('win_rate', 0):.2%}",
                    combo.get('trade_count', 0),
                    combo.get('winning_trades', 0),
                    combo.get('losing_trades', 0),
                    f"{combo.get('max_drawdown', 0):.4%}",
                    f"{combo.get('avg_hold_time', 0):.1f}",
                    f"{combo.get('avg_trade_return', 0):.4%}",
                    f"{combo.get('return_std', 0):.4f}",
                    f"${combo.get('end_taxed_liquidity', 0):,.2f}",
                    f"${combo.get('under1yearpl', 0):,.2f}",
                    f"${combo.get('over1yearpl', 0):,.2f}",
                    win_pct_last4_str
                )
                
                # Highlight best combination (original best) and highest score
                tags = []
                if combo == all_combinations[best_idx]:
                    tags.append("best")
                if i == 0:  # Highest score
                    tags.append("top_score")
                tree_cache.insert("", "end", values=values, tags=tuple(tags) if tags else ())
            
            # Configure tags for best combination and top score
            tree_cache.tag_configure("best", background="lightblue")
            tree_cache.tag_configure("top_score", background="lightgreen")
            
            # Bind selection to show trades
            tree_cache.bind("<<TreeviewSelect>>", show_trades_for_combination)
            
            # Auto-select best combination and show its trades
            if sorted_combos:
                first_item = tree_cache.get_children()[0] if tree_cache.get_children() else None
                if first_item:
                    tree_cache.selection_set(first_item)
                    tree_cache.focus(first_item)
                    show_trades_for_combination()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load cache file: {e}")
            import traceback
            traceback.print_exc()
    
    def export_cache_to_csv():
        """Export the current cache file to CSV."""
        selection = cache_combobox.current()
        if selection < 0 or selection >= len(current_cache_files):
            return
        
        cache_file = current_cache_files[selection]
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            all_combinations = cache_data.get('all_combinations', [])
            if not all_combinations:
                messagebox.showwarning("No Data", "No combinations to export.")
                return
            
            df = pd.DataFrame(all_combinations)
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Export Cache to CSV",
                initialfile=f"cache_{cache_data.get('ticker', 'unknown')}_{cache_data.get('end_date', 'unknown')}.csv"
            )
            
            if file_path:
                df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Cache exported to {file_path}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export cache: {e}")
    
    # Bind combobox selection
    cache_combobox.bind("<<ComboboxSelected>>", load_cache_file)
    
    # Load first file by default (if available)
    if current_cache_files:
        load_cache_file()
    
    def rescore_cache():
        """Rescore the current cache file with updated scoring configuration."""
        selection = cache_combobox.current()
        if selection < 0 or selection >= len(current_cache_files):
            return
        
        cache_file = current_cache_files[selection]
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Rescore using current scoring config
            rescored_data = scoring.rescore_cached_backtest(cache_data, scoring_config)
            
            # Reload the display
            load_cache_file()
            messagebox.showinfo("Success", "Cache rescored with current scoring configuration!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to rescore cache: {e}")
    
    # Buttons
    ttk.Button(bottom_frame, text="Refresh", command=load_cache_file).pack(side="left", padx=5)
    ttk.Button(bottom_frame, text="Rescore", command=rescore_cache).pack(side="left", padx=5)
    ttk.Button(bottom_frame, text="Export to CSV", command=export_cache_to_csv).pack(side="left", padx=5)

def add_custom_stock():
    """Add a custom stock symbol to the selection table."""
    custom_symbol = custom_stock_entry.get().strip().upper()
    if not custom_symbol:
        messagebox.showwarning("Warning", "Please enter a stock symbol.")
        return
    
    # Check if stock already exists in either table
    exists_in_main = any(
        tree.item(row)["values"][0] == custom_symbol
        for row in tree.get_children()
    )
    exists_in_custom = any(
        custom_tree.item(row)["values"][0] == custom_symbol
        for row in custom_tree.get_children()
    )
    
    if exists_in_main or exists_in_custom:
        messagebox.showwarning("Warning", f"Stock {custom_symbol} already exists in the table.")
        return
    
    try:
        # Try to fetch the stock name using yfinance
        import yfinance as yf
        stock = yf.Ticker(custom_symbol)
        info = stock.info
        if not info:
            raise ValueError("Could not fetch stock information")
        stock_name = info.get('longName', f"Custom Stock ({custom_symbol})")
        
        # Add to custom stocks table
        custom_tree.insert("", "end", values=(custom_symbol, stock_name))
        
        # Add to main selection table and handle mode-specific behavior
        tree_item = tree.insert("", "end", values=(custom_symbol, stock_name, ""), tags=("row",))
        
        # Handle mode-specific behavior
        global persistent_selected_stocks
        mode = mode_var.get()
        if mode == "Entire S&P 500" or mode == "Entire Russell 3000":
            # Automatically check all stocks
            for row in tree.get_children():
                tree.set(row, "Select", "✓")
            # Update persistent set to include all visible stocks
            persistent_selected_stocks = {tree.item(r)["values"][0] for r in tree.get_children()}
        elif mode == "Single Stock":
            # Clear all selections and select only the new stock
            for row in tree.get_children():
                tree.set(row, "Select", "")
            tree.set(tree_item, "Select", "✓")
            # Update persistent set - clear all and add only this one
            persistent_selected_stocks.clear()
            persistent_selected_stocks.add(custom_symbol)
        elif mode == "10 Stocks":
            # Count current selections
            selected_count = sum(1 for row in tree.get_children() if tree.set(row, "Select") == "✓")
            if selected_count < 10:
                tree.set(tree_item, "Select", "✓")
                # Update persistent set
                persistent_selected_stocks.add(custom_symbol)
        
        custom_stock_entry.delete(0, tk.END)  # Clear the entry field
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to add stock {custom_symbol}: {str(e)}")
        return

def create_ui():
    global root, mode_var, stock_data, tree, search_entry, tree_frame, search_frame
    global end_date_entry, log_text, progress_bar, status_label
    global time_frame_var, compounding_var, custom_stock_entry, custom_tree
    global show_visuals_var, visuals_checkbox, data, optimization_objective_var  # Add optimization_objective_var to globals
    global optimization_mapping, index_var  # Add optimization_mapping and index_var to globals
    global enable_walk_forward_var, backtest_years_var, backtest_months_var
    global walk_forward_years_var, walk_forward_months_var
    global timeframe_years_var, timeframe_months_var, timeframe_all_available_var

    # Check for library updates before proceeding
    check_library_updates()
    
    # Initialize with Russell 3000 (dynamic fetch on startup)
    stock_data = fetch_russell3000_tickers()

    root = tk.Tk()
    root.title("SMA Trading Simulation")
    root.geometry("1200x900")  # Increased width for better layout

    # Create main container frame
    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Index Selection Frame
    index_frame = ttk.LabelFrame(main_frame, text="Stock Index", padding="5")
    index_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
    index_var = tk.StringVar(value="Russell 3000")
    ttk.Radiobutton(index_frame, text="S&P 500", variable=index_var, value="S&P 500", command=on_index_change).grid(
        row=0, column=0, padx=20, pady=5
    )
    ttk.Radiobutton(index_frame, text="Russell 3000", variable=index_var, value="Russell 3000", command=on_index_change).grid(
        row=0, column=1, padx=20, pady=5
    )
    
    # Mode Selection Frame
    mode_frame = ttk.LabelFrame(main_frame, text="Trading Mode", padding="5")
    mode_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
    mode_var = tk.StringVar(value="Single Stock")
    modes = ["Single Stock", "10 Stocks", "Multi Select", "Entire S&P 500", "Entire Russell 3000"]
    for i, mode in enumerate(modes):
        ttk.Radiobutton(mode_frame, text=mode, variable=mode_var, value=mode, command=on_mode_change).grid(
            row=0, column=i, padx=15, pady=5
        )

    # Left Column Frame (Custom Stock + Search)
    left_frame = ttk.Frame(main_frame)
    left_frame.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")
    
    # Custom Stock Entry Frame
    custom_frame = ttk.LabelFrame(left_frame, text="Add Custom Stock", padding="5")
    custom_frame.pack(fill="x", pady=(0, 5))
    
    entry_frame = ttk.Frame(custom_frame)
    entry_frame.pack(fill="x", padx=5, pady=5)
    ttk.Label(entry_frame, text="Stock Symbol:").pack(side="left", padx=(0, 5))
    custom_stock_entry = ttk.Entry(entry_frame, width=15)
    custom_stock_entry.pack(side="left", padx=5)
    ttk.Button(entry_frame, text="Add Stock", command=add_custom_stock).pack(side="left", padx=5)
    
    # Custom stocks table with scrollbar
    custom_tree_frame = ttk.Frame(custom_frame)
    custom_tree_frame.pack(fill="x", padx=5, pady=5)
    
    custom_scroll = ttk.Scrollbar(custom_tree_frame)
    custom_scroll.pack(side="right", fill="y")
    
    custom_tree = ttk.Treeview(
        custom_tree_frame,
        columns=("Symbol", "Name"),
        show="headings",
        height=3,
        yscrollcommand=custom_scroll.set,
        selectmode="none"  # Prevent selection in the custom tree
    )
    custom_tree.heading("Symbol", text="Symbol")
    custom_tree.heading("Name", text="Name")
    custom_tree.column("Symbol", width=100)
    custom_tree.column("Name", width=250)
    custom_tree.pack(side="left", fill="x", expand=True)
    custom_scroll.config(command=custom_tree.yview)

    # Search Frame
    search_frame = ttk.LabelFrame(left_frame, text="Search Stocks", padding="5")
    search_frame.pack(fill="x")
    ttk.Label(search_frame, text="Search:").pack(side="left", padx=5)
    search_entry = ttk.Entry(search_frame, width=40)
    search_entry.pack(side="left", padx=5, pady=5, fill="x", expand=True)
    search_entry.bind("<KeyRelease>", lambda e: filter_table(search_entry.get()))

    # Right Column Frame (Settings)
    right_frame = ttk.Frame(main_frame)
    right_frame.grid(row=2, column=1, padx=5, pady=5, sticky="nsew")
    
    # Settings Frame
    settings_frame = ttk.LabelFrame(right_frame, text="Settings", padding="5")
    settings_frame.pack(fill="x")
    
    # Date and Time Frame settings
    date_frame = ttk.Frame(settings_frame)
    date_frame.pack(fill="x", pady=5)
    ttk.Label(date_frame, text="End Date:").pack(side="left", padx=5)
    end_date_entry = ttk.Entry(date_frame, width=15)
    end_date_entry.pack(side="left", padx=5)
    end_date_entry.insert(0, (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"))

    time_frame = ttk.Frame(settings_frame)
    time_frame.pack(fill="x", pady=5)
    ttk.Label(time_frame, text="Time Frame:").pack(side="left", padx=5)
    
    # All Available checkbox
    timeframe_all_available_var = tk.BooleanVar(value=False)
    timeframe_all_checkbox = ttk.Checkbutton(time_frame, text="All Available", variable=timeframe_all_available_var)
    timeframe_all_checkbox.pack(side="left", padx=5)
    
    # Year and Month inputs (similar to walk forward)
    timeframe_years_var = tk.IntVar(value=5)
    timeframe_months_var = tk.IntVar(value=0)
    ttk.Spinbox(time_frame, from_=0, to=50, textvariable=timeframe_years_var, width=5).pack(side="left", padx=2)
    ttk.Label(time_frame, text="Y").pack(side="left", padx=1)
    ttk.Spinbox(time_frame, from_=0, to=11, textvariable=timeframe_months_var, width=5).pack(side="left", padx=2)
    ttk.Label(time_frame, text="M").pack(side="left", padx=1)
    
    # Keep time_frame_var for backward compatibility (not used anymore)
    time_frame_var = tk.StringVar(value="Custom")

    # Optimization Objective Frame
    optimization_frame = ttk.Frame(settings_frame)
    optimization_frame.pack(fill="x", pady=5)
    ttk.Label(optimization_frame, text="Optimization:").pack(side="left", padx=5)
    optimization_objective_var = tk.StringVar(value="taxed_return")
    optimization_options = ["taxed_return", "better_off", "win_rate"]
    optimization_labels = ["Taxed Return", "Better Off", "Win Rate"]
    optimization_combobox = ttk.Combobox(optimization_frame, textvariable=optimization_objective_var, values=optimization_labels, state="readonly", width=15)
    optimization_combobox.pack(side="left", padx=5)
    
    # Map display labels to internal values
    optimization_mapping = {
        "Taxed Return": "taxed_return",
        "Better Off": "better_off", 
        "Win Rate": "win_rate"
    }
    
    # Set initial value
    optimization_objective_var.set("Taxed Return")

    # Options Frame
    options_frame = ttk.Frame(settings_frame)
    options_frame.pack(fill="x", pady=5)
    
    compounding_var = tk.BooleanVar(value=True)
    compounding_checkbox = ttk.Checkbutton(
        options_frame, 
        text="Compound Gains",
        variable=compounding_var
    )
    compounding_checkbox.pack(side="left", padx=5)

    show_visuals_var = tk.BooleanVar(value=False)
    visuals_checkbox = ttk.Checkbutton(
        options_frame,
        text="Show Interactive Charts",
        variable=show_visuals_var,
        state="normal" if mode_var.get() == "Single Stock" else "disabled"
    )
    visuals_checkbox.pack(side="left", padx=20)
    
    # Walk-Forward Analysis Frame
    walk_forward_frame = ttk.LabelFrame(settings_frame, text="Walk-Forward Analysis (Optional)", padding="5")
    walk_forward_frame.pack(fill="x", pady=5)
    
    # Enable Walk-Forward toggle
    enable_walk_forward_var = tk.BooleanVar(value=False)
    enable_walk_forward_checkbox = ttk.Checkbutton(
        walk_forward_frame,
        text="Enable Walk-Forward Analysis",
        variable=enable_walk_forward_var
    )
    enable_walk_forward_checkbox.pack(anchor="w", pady=2)
    
    # Walk-forward inputs container
    wf_inputs_frame = ttk.Frame(walk_forward_frame)
    wf_inputs_frame.pack(fill="x", pady=5)
    
    # Backtest Period (training period)
    backtest_period_frame = ttk.Frame(wf_inputs_frame)
    backtest_period_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=2)
    ttk.Label(backtest_period_frame, text="Backtest Period:", width=18, anchor="w").pack(side="left", padx=2)
    backtest_years_var = tk.IntVar(value=4)
    backtest_months_var = tk.IntVar(value=0)
    ttk.Spinbox(backtest_period_frame, from_=0, to=20, textvariable=backtest_years_var, width=5).pack(side="left", padx=2)
    ttk.Label(backtest_period_frame, text="Y").pack(side="left", padx=1)
    ttk.Spinbox(backtest_period_frame, from_=0, to=11, textvariable=backtest_months_var, width=5).pack(side="left", padx=2)
    ttk.Label(backtest_period_frame, text="M").pack(side="left", padx=1)
    
    # Walk-Forward Period (testing period)
    walk_forward_period_frame = ttk.Frame(wf_inputs_frame)
    walk_forward_period_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=2)
    ttk.Label(walk_forward_period_frame, text="Walk-Forward Period:", width=18, anchor="w").pack(side="left", padx=2)
    walk_forward_years_var = tk.IntVar(value=1)
    walk_forward_months_var = tk.IntVar(value=0)
    ttk.Spinbox(walk_forward_period_frame, from_=0, to=10, textvariable=walk_forward_years_var, width=5).pack(side="left", padx=2)
    ttk.Label(walk_forward_period_frame, text="Y").pack(side="left", padx=1)
    ttk.Spinbox(walk_forward_period_frame, from_=0, to=11, textvariable=walk_forward_months_var, width=5).pack(side="left", padx=2)
    ttk.Label(walk_forward_period_frame, text="M").pack(side="left", padx=1)
    
    wf_inputs_frame.columnconfigure(0, weight=1)

    # Stock Selection Frame
    selection_frame = ttk.LabelFrame(main_frame, text="Stock Selection", padding="5")
    selection_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
    
    # Create a container for search and tree
    selection_container = ttk.Frame(selection_frame)
    selection_container.pack(fill="both", expand=True, padx=5, pady=5)
    
    # Tree Frame
    tree_frame = ttk.Frame(selection_container)
    tree_frame.pack(fill="both", expand=True)
    
    tree_scroll_y = ttk.Scrollbar(tree_frame)
    tree_scroll_y.pack(side="right", fill="y")
    tree_scroll_x = ttk.Scrollbar(tree_frame, orient="horizontal")
    tree_scroll_x.pack(side="bottom", fill="x")

    tree = ttk.Treeview(
        tree_frame, 
        columns=("Symbol", "Name", "Select"), 
        show="headings",
        yscrollcommand=tree_scroll_y.set,
        xscrollcommand=tree_scroll_x.set,
        height=15  # Increased height for better visibility
    )
    tree_scroll_y.config(command=tree.yview)
    tree_scroll_x.config(command=tree.xview)
    tree.heading("Symbol", text="Ticker")
    tree.heading("Name", text="Name")
    tree.heading("Select", text="Select")
    tree.column("Symbol", width=100)
    tree.column("Name", width=400)
    tree.column("Select", width=50, anchor="center")
    tree.bind("<Button-1>", toggle_checkbox)
    tree.pack(fill="both", expand=True)
    
    # Initial population of the table
    filter_table("")

    # Button Frame
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=4, column=0, columnspan=2, pady=10)
    
    style = ttk.Style()
    style.configure("Action.TButton", padding=5)
    style.configure("Cancel.TButton", padding=5)
    
    global run_button, cancel_button
    run_button = ttk.Button(button_frame, text="Run Now", command=on_run_now, style="Action.TButton")
    run_button.pack(side="left", padx=10)
    
    # Create a frame for the cancel button to give it a red background
    cancel_frame = ttk.Frame(button_frame)
    cancel_frame.pack(side="left", padx=10)
    cancel_frame.configure(style="Cancel.TFrame")
    
    cancel_button = ttk.Button(
        cancel_frame, 
        text="Cancel", 
        command=request_cancel, 
        style="Cancel.TButton",
        state="disabled"
    )
    cancel_button.pack()
    
    trade_table_button = ttk.Button(button_frame, text="Look at Trade Table", command=view_trade_table, style="Action.TButton")
    trade_table_button.pack(side="left", padx=10)
    
    export_button = ttk.Button(button_frame, text="Export Results to CSV", command=export_results_to_csv, style="Action.TButton")
    export_button.pack(side="left", padx=10)
    
    view_cache_button = ttk.Button(button_frame, text="View Cached Backtests", command=view_cached_backtests, style="Action.TButton")
    view_cache_button.pack(side="left", padx=10)
    
    scoring_config_button = ttk.Button(button_frame, text="Scoring Config", command=open_scoring_config, style="Action.TButton")
    scoring_config_button.pack(side="left", padx=10)
    
    def view_or_load_batch():
        """Smart batch view - shows current results if available, otherwise prompts to load from folder."""
        global algorithm_results
        
        # Check if we have current results with multiple stocks
        if algorithm_results:
            valid_results = {k: v for k, v in algorithm_results.items() if "Error" not in v}
            if len(valid_results) >= 2:
                # We have a batch - show it
                view_batch_results()
                return
        
        # No current batch - offer to load from folder
        response = messagebox.askyesno(
            "No Current Batch Results",
            "No batch results are currently available.\n\n"
            "Would you like to load a batch from a saved folder?\n\n"
            "Click 'Yes' to select a batch folder, or 'No' to cancel."
        )
        if response:
            load_batch_from_folder()
    
    batch_view_button = ttk.Button(button_frame, text="View Batch Results", command=view_or_load_batch, style="Action.TButton")
    batch_view_button.pack(side="left", padx=10)

    # Log Frame
    log_frame = ttk.LabelFrame(main_frame, text="Logs", padding="5")
    log_frame.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
    
    log_text = tk.Text(log_frame, height=10, width=120, wrap="word")
    log_text.pack(fill="both", expand=True, padx=5, pady=5)

    # Status Bar
    status_frame = ttk.Frame(main_frame)
    status_frame.grid(row=6, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
    
    status_label = ttk.Label(status_frame, text="Ready", font=("Arial", 10))
    status_label.pack(side="left")
    
    progress_bar = ttk.Progressbar(status_frame, mode="determinate", maximum=100)
    progress_bar.pack(side="right", fill="x", expand=True, padx=(10, 0))

    # Configure grid weights
    main_frame.columnconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)
    main_frame.rowconfigure(3, weight=1)
    main_frame.rowconfigure(5, weight=1)

    root.mainloop()

if __name__ == "__main__":
    create_ui()

