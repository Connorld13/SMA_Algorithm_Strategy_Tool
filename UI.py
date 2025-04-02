# UI.py

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import threading

from fetchdata import fetch_and_prepare_data
import algorithm

# ---- NEW IMPORT ----
import visual  # <-- (1) Import the new visual.py

# Global variable to store algorithm results
algorithm_results = {}

# Declare global widgets to be accessed by set_status and update_progress
root = None
status_label = None
progress_bar = None

# Add new global variable for cancel flag
cancel_requested = False

def fetch_sp500_tickers_from_csv():
    """Fetch S&P 500 tickers and names from a local CSV."""
    try:
        sp500_data = pd.read_csv("sp500_companies.csv")
        return sp500_data
    except Exception as e:
        messagebox.showerror("Error", f"Failed to fetch S&P 500 tickers from CSV: {e}")
        return pd.DataFrame(columns=["Symbol", "Name"])

def on_mode_change():
    """Update UI based on the selected mode."""
    global visuals_checkbox  # Add global declaration
    mode = mode_var.get()
    search_entry.delete(0, tk.END)
    filter_table("")

    # Always show the stock selection table and search frame
    # These are now managed by pack, so no need to show/hide them
    
    if mode == "Entire S&P 500":
        # Automatically check all stocks
        for row in tree.get_children():
            tree.set(row, "Select", "✓")
    else:
        # Clear all selections when switching
        for row in tree.get_children():
            tree.set(row, "Select", "")
            
    # Update the visuals checkbox state
    if mode == "Single Stock":
        visuals_checkbox.configure(state="normal")
    else:
        show_visuals_var.set(False)
        visuals_checkbox.configure(state="disabled")

def toggle_checkbox(event):
    """Toggle the checkbox state in the Treeview."""
    mode = mode_var.get()
    region = tree.identify_region(event.x, event.y)
    if region == "cell":
        row_id = tree.identify_row(event.y)
        col_id = tree.identify_column(event.x)
        if col_id == "#3":  # Checkbox column
            if mode == "Entire S&P 500":
                # Prevent unchecking in Entire S&P 500 mode
                messagebox.showinfo("Info", "All stocks are automatically selected in Entire S&P 500 mode.")
                return
            current_value = tree.set(row_id, "Select")
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
            elif mode == "10 Stocks":
                if len(selected_stocks) >= 10 and current_value == "":
                    messagebox.showwarning("Limit Reached", "You can only select up to 10 stocks.")
                else:
                    new_value = "✓" if current_value == "" else ""
                    tree.set(row_id, "Select", new_value)
            else:
                # For "Multi Select" and other modes without limits
                new_value = "✓" if current_value == "" else ""
                tree.set(row_id, "Select", new_value)

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
    if mode == "Entire S&P 500":
        # Ensure all stocks are selected
        all_stocks = [tree.item(r)["values"][0] for r in tree.get_children()]
        if set(selected_stocks) != set(all_stocks):
            messagebox.showerror("Error", "All stocks must be selected for Entire S&P 500 mode.")
            return None
    return selected_stocks

def filter_table(query):
    """Filter the table based on the search query."""
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
            tree.insert("", "end", values=(symbol, name, ""), tags=("row",))
    
    # Then add S&P 500 stocks
    if sp500_data is not None:
        for _, row_data in sp500_data.iterrows():
            if (query.lower() in str(row_data["Symbol"]).lower() or 
                query.lower() in str(row_data["Name"]).lower()):
                # Skip if this stock is already added as custom
                if row_data["Symbol"] not in custom_stocks:
                    tree.insert("", "end", values=(row_data["Symbol"], row_data["Name"], ""), tags=("row",))

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

            # Get selected time frame
            time_frame = time_frame_var.get()

            # Decide if we are compounding or not
            is_compounding = compounding_var.get()

            # Calculate start_date or None + num_rows
            start_date_str = None
            num_rows = None

            if time_frame == "All Available":
                start_date_str = None
                num_rows = None
            else:
                if time_frame == "10 Years":
                    start_date_dt = end_date_dt - relativedelta(years=10)
                    num_rows = 252 * 10
                elif time_frame == "5 Years":
                    start_date_dt = end_date_dt - relativedelta(years=5)
                    num_rows = 252 * 5
                elif time_frame == "3 Years":
                    start_date_dt = end_date_dt - relativedelta(years=3)
                    num_rows = 252 * 3
                elif time_frame == "1 Year":
                    start_date_dt = end_date_dt - relativedelta(years=1)
                    num_rows = 252
                elif time_frame == "6 Months":
                    start_date_dt = end_date_dt - relativedelta(months=6)
                    num_rows = 128
                elif time_frame == "3 Months":
                    start_date_dt = end_date_dt - relativedelta(months=3)
                    num_rows = 64
                elif time_frame == "1 Month":
                    start_date_dt = end_date_dt - relativedelta(months=1)
                    num_rows = 22
                else:
                    # Fallback: 5 Years
                    start_date_dt = end_date_dt - relativedelta(years=5)
                    num_rows = 252 * 5

                start_date_str = start_date_dt.strftime("%Y-%m-%d")

            # Determine selected stocks
            if mode == "Entire S&P 500":
                selected_stocks = sp500_data['Symbol'].tolist()
                # Ensure all checkboxes are checked
                for row in tree.get_children():
                    tree.set(row, "Select", "✓")
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
            data = fetch_and_prepare_data(selected_stocks, start_date_str, end_date_str, num_rows)
            update_progress(20)

            if cancel_requested:
                set_status("Operation cancelled.")
                return

            # Run algorithm with more frequent cancellation checks
            set_status("Running algorithm...")
            total_algorithm_progress = 80
            per_stock_progress = total_algorithm_progress / total_selected if total_selected > 0 else total_algorithm_progress

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
                    result = algorithm.run_algorithm(
                        ticker_data,
                        start_amount=10000,
                        progress_callback=progress_callback,
                        compounding=is_compounding
                    )
                    algorithm_results[ticker] = result
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
    global data  # Add global declaration
    if cancel_requested:
        return
        
    log_text.delete(1.0, tk.END)
    for ticker, result in algorithm_results.items():
        log_text.insert(tk.END, f"Results for {ticker}:\n")
        if "Error" in result:
            log_text.insert(tk.END, f"  {result['Error']}\n\n")
            continue
        log_text.insert(tk.END, "  Output Results 1:\n")
        for key, value in result['outputresults1'].items():
            log_text.insert(tk.END, f"    {key}: {value}\n")
        log_text.insert(tk.END, "  Output Results 2:\n")
        for key, value in result['outputresults2'].items():
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
    """Open a new window to display the best trades for selected tickers."""
    if not algorithm_results:
        messagebox.showwarning("No Data", "Please run the algorithm first to view trade tables.")
        return

    trade_window = tk.Toplevel(root)
    trade_window.title("Best Trades Table")
    trade_window.geometry("1000x600")

    ttk.Label(trade_window, text="Select Ticker:", font=("Arial", 12)).pack(pady=10)
    ticker_var = tk.StringVar()
    tickers = list(algorithm_results.keys())
    ticker_combobox = ttk.Combobox(trade_window, textvariable=ticker_var, values=tickers, state="readonly", width=20)
    ticker_combobox.pack(pady=5)

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

    def populate_trade_table(event=None):
        selected_ticker = ticker_var.get()
        if not selected_ticker:
            return

        for item in tree_trade.get_children():
            tree_trade.delete(item)
        tree_trade["columns"] = ()
        tree_trade["show"] = "headings"

        result = algorithm_results[selected_ticker]
        if "Error" in result:
            messagebox.showerror("Error", f"No trades to display for {selected_ticker}.")
            return

        best_trades = result.get("besttrades", [])
        if not best_trades:
            messagebox.showinfo("No Trades", f"No trades were generated for {selected_ticker}.")
            return

        columns = list(best_trades[0].keys())
        tree_trade["columns"] = columns
        for col in columns:
            tree_trade.heading(col, text=col)
            tree_trade.column(col, anchor="center", width=120)

        for trade in best_trades:
            values = [trade[col] for col in columns]
            tree_trade.insert("", "end", values=values)

    ticker_combobox.bind("<<ComboboxSelected>>", populate_trade_table)

def export_results_to_csv():
    """Export the algorithm results to a CSV file with the specified format."""
    if not algorithm_results:
        messagebox.showwarning("No Data", "Please run the algorithm first before exporting results.")
        return

    export_data = []
    today_str = datetime.now().strftime("%Y-%m-%d")
    strategy = "sma2024"
    total_timeframe = time_frame_var.get()

    for symbol, result in algorithm_results.items():
        if "Error" in result:
            continue

        output1 = result.get("outputresults1", {})
        output2 = result.get("outputresults2", {})

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

        export_row = {
            "Symbol": symbol,
            "Test Date": today_str,
            "strategy": strategy,
            "Total Timeframe": total_timeframe,
            "Input 1": besta,
            "Input 2": bestb,
            "5 year diff - %": betteroff,
            "Taxed Return": besttaxedreturn,
            "NoAlgoReturn": noalgoreturn,
            "Win Rate": winningtradepct,
            "Max Drawdown": maxdrawdown,
            "# of Closed Trades": besttradecount,
            "Avg Hold Time": avg_hold_time,
            "Win % Last 4 Trades": win_pct_last_4
        }

        export_data.append(export_row)

    export_df = pd.DataFrame(export_data, columns=[
        "Symbol",
        "Test Date",
        "strategy",
        "Total Timeframe",
        "Input 1",
        "Input 2",
        "5 year diff - %",
        "Taxed Return",
        "NoAlgoReturn",
        "Win Rate",
        "Max Drawdown",
        "# of Closed Trades",
        "Avg Hold Time",
        "Win % Last 4 Trades"
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
        mode = mode_var.get()
        if mode == "Entire S&P 500":
            # Automatically check all stocks
            for row in tree.get_children():
                tree.set(row, "Select", "✓")
        elif mode == "Single Stock":
            # Clear all selections and select only the new stock
            for row in tree.get_children():
                tree.set(row, "Select", "")
            tree.set(tree_item, "Select", "✓")
        elif mode == "10 Stocks":
            # Count current selections
            selected_count = sum(1 for row in tree.get_children() if tree.set(row, "Select") == "✓")
            if selected_count < 10:
                tree.set(tree_item, "Select", "✓")
        
        custom_stock_entry.delete(0, tk.END)  # Clear the entry field
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to add stock {custom_symbol}: {str(e)}")
        return

def create_ui():
    global root, mode_var, sp500_data, tree, search_entry, tree_frame, search_frame
    global end_date_entry, log_text, progress_bar, status_label
    global time_frame_var, compounding_var, custom_stock_entry, custom_tree
    global show_visuals_var, visuals_checkbox, data  # Add data to globals

    sp500_data = fetch_sp500_tickers_from_csv()

    root = tk.Tk()
    root.title("SMA Trading Simulation")
    root.geometry("1200x900")  # Increased width for better layout

    # Create main container frame
    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Mode Selection Frame
    mode_frame = ttk.LabelFrame(main_frame, text="Trading Mode", padding="5")
    mode_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
    mode_var = tk.StringVar(value="Single Stock")
    modes = ["Single Stock", "10 Stocks", "Multi Select", "Entire S&P 500"]
    for i, mode in enumerate(modes):
        ttk.Radiobutton(mode_frame, text=mode, variable=mode_var, value=mode, command=on_mode_change).grid(
            row=0, column=i, padx=20, pady=5
        )

    # Left Column Frame (Custom Stock + Search)
    left_frame = ttk.Frame(main_frame)
    left_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
    
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
    right_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
    
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
    time_frame_var = tk.StringVar(value="5 Years")
    time_frame_options = ["All Available", "10 Years", "5 Years", "3 Years", "1 Year", "6 Months", "3 Months", "1 Month"]
    time_frame_combobox = ttk.Combobox(time_frame, textvariable=time_frame_var, values=time_frame_options, state="readonly", width=15)
    time_frame_combobox.pack(side="left", padx=5)

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

    show_visuals_var = tk.BooleanVar(value=True)
    visuals_checkbox = ttk.Checkbutton(
        options_frame,
        text="Show Interactive Charts",
        variable=show_visuals_var,
        state="normal" if mode_var.get() == "Single Stock" else "disabled"
    )
    visuals_checkbox.pack(side="left", padx=20)

    # Stock Selection Frame
    selection_frame = ttk.LabelFrame(main_frame, text="Stock Selection", padding="5")
    selection_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
    
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
    button_frame.grid(row=3, column=0, columnspan=2, pady=10)
    
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

    # Log Frame
    log_frame = ttk.LabelFrame(main_frame, text="Logs", padding="5")
    log_frame.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
    
    log_text = tk.Text(log_frame, height=10, width=120, wrap="word")
    log_text.pack(fill="both", expand=True, padx=5, pady=5)

    # Status Bar
    status_frame = ttk.Frame(main_frame)
    status_frame.grid(row=5, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
    
    status_label = ttk.Label(status_frame, text="Ready", font=("Arial", 10))
    status_label.pack(side="left")
    
    progress_bar = ttk.Progressbar(status_frame, mode="determinate", maximum=100)
    progress_bar.pack(side="right", fill="x", expand=True, padx=(10, 0))

    # Configure grid weights
    main_frame.columnconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)
    main_frame.rowconfigure(2, weight=1)
    main_frame.rowconfigure(4, weight=1)

    root.mainloop()

if __name__ == "__main__":
    create_ui()
