# UI.py

import tkinter as tk
from tkinter import ttk, messagebox, filedialog  # Added filedialog
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import threading
from fetchdata import fetch_and_prepare_data
import algorithm  # Import the refactored algorithm module

# Global variable to store algorithm results
algorithm_results = {}

# Declare global widgets to be accessed by set_status and update_progress
root = None
status_label = None
progress_bar = None

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
    mode = mode_var.get()
    search_entry.delete(0, tk.END)
    filter_table("")
    
    # **Always show the stock selection table**
    tree_frame.grid()
    search_frame.grid()

    if mode == "Entire S&P 500":
        # **Automatically check all stocks**
        for row in tree.get_children():
            tree.set(row, "Select", "✓")
    else:
        # **Clear all selections when switching to other modes**
        for row in tree.get_children():
            tree.set(row, "Select", "")

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
                tree.item(row)["values"][0]
                for row in tree.get_children()
                if tree.set(row, "Select") == "✓"
            ]

            if mode == "Single Stock" and len(selected_stocks) >= 1 and current_value == "":
                # Allow only one checkbox for Single Stock
                for row in tree.get_children():
                    tree.set(row, "Select", "")
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
    selected_stocks = [tree.item(row)["values"][0] for row in tree.get_children() if tree.set(row, "Select") == "✓"]
    
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
        all_stocks = [tree.item(row)["values"][0] for row in tree.get_children()]
        if set(selected_stocks) != set(all_stocks):
            messagebox.showerror("Error", "All stocks must be selected for Entire S&P 500 mode.")
            return None
    return selected_stocks

def filter_table(query):
    """Filter the table based on the search query."""
    for row in tree.get_children():
        tree.delete(row)
    for _, row in sp500_data.iterrows():
        if query.lower() in row["Symbol"].lower() or query.lower() in row["Name"].lower():
            tree.insert("", "end", values=(row["Symbol"], row["Name"], ""), tags=("row",))

def set_status(text):
    """Update the status label in a thread-safe manner."""
    if root and status_label:
        root.after(0, lambda: status_label.config(text=text))

def update_progress(value):
    """Update the progress bar."""
    if progress_bar:
        progress_bar["value"] = value
        root.update_idletasks()  # Force UI update

def on_run_now():
    """Fetch data and execute the strategy in a separate thread with progress tracking."""
    def run_algorithm_thread():
        global algorithm_results
        try:
            # Reset and start progress bar
            set_status("Starting algorithm...")
            update_progress(0)

            mode = mode_var.get()

            # Get end date input
            end_date = end_date_entry.get()

            # Validate end date
            try:
                end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
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

            # Calculate start date as 5 years before end date
            start_date_dt = end_date_dt - relativedelta(years=5)
            start_date = start_date_dt.strftime("%Y-%m-%d")

            # Determine selected stocks based on mode
            if mode == "Entire S&P 500":
                selected_stocks = sp500_data['Symbol'].tolist()
                # Ensure all checkboxes are checked
                for row in tree.get_children():
                    tree.set(row, "Select", "✓")
            else:
                # Validate stock selection
                selected_stocks = validate_stock_selection()
                if selected_stocks is None:
                    update_progress(0)
                    set_status("Ready")
                    return

            total_selected = len(selected_stocks)

            # Fetch data
            set_status("Fetching data...")
            # Data fetching is considered as 20% of the progress
            data = fetch_and_prepare_data(selected_stocks, start_date, end_date)
            update_progress(20)

            # Run algorithm
            set_status("Running algorithm...")
            # Allocate 80% of the progress bar to algorithm execution
            total_algorithm_progress = 80
            per_stock_progress = total_algorithm_progress / total_selected if total_selected > 0 else total_algorithm_progress

            algorithm_results = {}
            for idx, ticker in enumerate(selected_stocks, start=1):
                current_stock_index = idx  # 1-based index

                def progress_callback(algorithm_progress, current_stock_index=current_stock_index):
                    """Callback to update progress based on algorithm's progress."""
                    # Calculate overall progress:
                    # 20% for data fetching + (current_stock_index - 1) * per_stock_progress + algorithm_progress% of per_stock_progress
                    overall_progress = 20 + (current_stock_index - 1) * per_stock_progress + (algorithm_progress / 100) * per_stock_progress
                    overall_progress = min(overall_progress, 100)  # Ensure it does not exceed 100
                    update_progress(overall_progress)

                set_status(f"Running algorithm on stock {idx}/{total_selected} ({ticker})...")
                ticker_data = data[data['Ticker'] == ticker].copy()
                if ticker_data.empty:
                    algorithm_results[ticker] = {"Error": "No data fetched for this ticker."}
                    # Immediately update progress for this stock
                    progress_callback(100)
                else:
                    result = algorithm.run_algorithm(ticker_data, start_amount=10000, progress_callback=progress_callback)
                    algorithm_results[ticker] = result
                    # Ensure progress is updated to 100% for this stock
                    progress_callback(100)

            # Finalize progress
            update_progress(100)
            set_status("Finalizing...")

            # Display results in log text (excluding best trades)
            log_text.delete(1.0, tk.END)  # Clear previous logs
            for ticker, result in algorithm_results.items():
                log_text.insert(tk.END, f"Results for {ticker}:\n")
                if "Error" in result:
                    log_text.insert(tk.END, f"  {result['Error']}\n\n")
                    continue
                # Display Output Results 1
                log_text.insert(tk.END, "  Output Results 1:\n")
                for key, value in result['outputresults1'].items():
                    log_text.insert(tk.END, f"    {key}: {value}\n")
                # Display Output Results 2
                log_text.insert(tk.END, "  Output Results 2:\n")
                for key, value in result['outputresults2'].items():
                    log_text.insert(tk.END, f"    {key}: {value}\n")
                log_text.insert(tk.END, "\n")

            # Inform the user that results are ready
            messagebox.showinfo("Success", "Algorithm execution completed. You can view the trade tables using the 'Look at Trade Table' button.")
            set_status("Completed.")
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            update_progress(0)
            set_status("Ready")
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error: {e}")
            update_progress(0)
            set_status("Ready")
        finally:
            update_progress(100)  # Ensure progress bar completes at the end

    # Run the algorithm in a separate thread
    thread = threading.Thread(target=run_algorithm_thread)
    thread.start()

def view_trade_table():
    """Open a new window to display the best trades for selected tickers."""
    if not algorithm_results:
        messagebox.showwarning("No Data", "Please run the algorithm first to view trade tables.")
        return

    # Create a new top-level window
    trade_window = tk.Toplevel(root)
    trade_window.title("Best Trades Table")
    trade_window.geometry("1000x600")  # Adjust size as needed

    # Dropdown to select ticker
    ttk.Label(trade_window, text="Select Ticker:", font=("Arial", 12)).pack(pady=10)
    ticker_var = tk.StringVar()
    tickers = list(algorithm_results.keys())
    ticker_combobox = ttk.Combobox(trade_window, textvariable=ticker_var, values=tickers, state="readonly", width=20)
    ticker_combobox.pack(pady=5)

    # Frame for Treeview
    tree_frame_trade = ttk.Frame(trade_window)
    tree_frame_trade.pack(fill="both", expand=True, padx=10, pady=10)

    # Scrollbars for Treeview
    tree_scroll_y = ttk.Scrollbar(tree_frame_trade, orient="vertical")
    tree_scroll_y.pack(side="right", fill="y")
    tree_scroll_x = ttk.Scrollbar(tree_frame_trade, orient="horizontal")
    tree_scroll_x.pack(side="bottom", fill="x")

    # Treeview widget
    tree_trade = ttk.Treeview(tree_frame_trade, yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
    tree_trade.pack(fill="both", expand=True)
    tree_scroll_y.config(command=tree_trade.yview)
    tree_scroll_x.config(command=tree_trade.xview)

    def populate_trade_table(event=None):
        """Populate the Treeview with the best trades for the selected ticker."""
        selected_ticker = ticker_var.get()
        if not selected_ticker:
            return

        # Clear existing data
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

        # Create columns based on DataFrame keys
        columns = list(best_trades[0].keys())
        tree_trade["columns"] = columns
        for col in columns:
            tree_trade.heading(col, text=col)
            # Adjust column width as needed
            if col == 'Date':
                tree_trade.column(col, anchor="center", width=100)
            elif col == 'Price':
                tree_trade.column(col, anchor="center", width=80)
            else:
                tree_trade.column(col, anchor="center", width=120)

        # Insert data into Treeview
        for trade in best_trades:
            values = [trade[col] for col in columns]
            tree_trade.insert("", "end", values=values)

    # Bind the selection event
    ticker_combobox.bind("<<ComboboxSelected>>", populate_trade_table)

def export_results_to_csv():
    """Export the algorithm results to a CSV file with the specified format."""
    if not algorithm_results:
        messagebox.showwarning("No Data", "Please run the algorithm first before exporting results.")
        return

    # Prepare the data
    export_data = []
    today_str = datetime.now().strftime("%Y-%m-%d")
    strategy = "sma2024"
    total_timeframe = "5 years"

    for symbol, result in algorithm_results.items():
        if "Error" in result:
            # Skip symbols with errors or handle accordingly
            continue

        output1 = result.get("outputresults1", {})
        output2 = result.get("outputresults2", {})

        # Extract required fields with default values if keys are missing
        besta = output1.get("besta", "")
        bestb = output1.get("bestb", "")
        betteroff = output1.get("betteroff", 0) * 100  # Convert to percentage
        besttaxedreturn = output1.get("besttaxedreturn", 0) * 100  # Convert to percentage
        noalgoreturn = output1.get("noalgoreturn", 0) * 100  # Convert to percentage
        losingtradepct = output2.get("losingtradepct", 0)
        win_rate = (losingtradepct - 1) * 100  # As per user instruction
        maxdrawdown = output2.get("maxdrawdown(worst trade return pct)", 0) * 100  # **Multiply by 100**
        besttradecount = output1.get("besttradecount", 0)

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
            "Win Rate": win_rate,
            "Max Drawdown": maxdrawdown,  # **Already multiplied by 100**
            "# of Closed Trades": besttradecount
        }

        export_data.append(export_row)

    # Convert to DataFrame
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
        "# of Closed Trades"
    ])

    if export_df.empty:
        messagebox.showinfo("No Data", "There are no results to export.")
        return

    # Ask user where to save the CSV
    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        title="Save Results As"
    )

    if not file_path:
        # User cancelled the save dialog
        return

    try:
        export_df.to_csv(file_path, index=False)
        messagebox.showinfo("Success", f"Results successfully exported to {file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save CSV file: {e}")

def create_ui():
    # Default end date: yesterday
    yesterday_dt = datetime.now() - timedelta(days=1)
    yesterday_str = yesterday_dt.strftime("%Y-%m-%d")

    # Calculate start date as 5 years before yesterday
    start_date_dt = yesterday_dt - relativedelta(years=5)
    start_date_str = start_date_dt.strftime("%Y-%m-%d")  # For reference, not used directly

    global root, mode_var, sp500_data, tree, search_entry, tree_frame, search_frame
    global end_date_entry, log_text, progress_bar, status_label  # Declare progress_bar and status_label as global

    # Load S&P 500 tickers
    sp500_data = fetch_sp500_tickers_from_csv()

    # Main Window
    root = tk.Tk()
    root.title("SMA Trading Simulation")
    root.geometry("1000x850")  # Increased width and height to accommodate trade table button and status label

    # Mode Selection
    mode_var = tk.StringVar(value="Single Stock")
    ttk.Label(root, text="Select Mode:", font=("Arial", 12)).grid(row=0, column=0, padx=10, pady=10, sticky="w")
    # Updated modes list: Removed "Half of S&P 500", added "Multi Select"
    modes = ["Entire S&P 500", "10 Stocks", "Single Stock", "Multi Select"]
    for i, mode in enumerate(modes):
        ttk.Radiobutton(root, text=mode, variable=mode_var, value=mode, command=on_mode_change).grid(row=0, column=i + 1, padx=10, pady=10)

    # Search Bar
    search_frame = ttk.LabelFrame(root, text="Search Stocks")
    search_frame.grid(row=1, column=0, columnspan=5, padx=10, pady=5, sticky="ew")
    search_label = ttk.Label(search_frame, text="Search:")
    search_label.grid(row=0, column=0, padx=5, pady=5)
    search_entry = ttk.Entry(search_frame, width=30)
    search_entry.grid(row=0, column=1, padx=5, pady=5)
    search_entry.bind("<KeyRelease>", lambda e: filter_table(search_entry.get()))

    # Stock Table
    tree_frame = ttk.LabelFrame(root, text="Stock Selection")
    tree_frame.grid(row=2, column=0, columnspan=5, padx=10, pady=5, sticky="ew")
    tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical")
    tree_scroll.pack(side="right", fill="y")
    tree_scroll_x = ttk.Scrollbar(tree_frame, orient="horizontal")
    tree_scroll_x.pack(side="bottom", fill="x")
    tree = ttk.Treeview(tree_frame, columns=("Symbol", "Name", "Select"), show="headings", yscrollcommand=tree_scroll.set, xscrollcommand=tree_scroll_x.set)
    tree_scroll.config(command=tree.yview)
    tree_scroll_x.config(command=tree.xview)
    tree.heading("Symbol", text="Ticker")
    tree.heading("Name", text="Name")
    tree.heading("Select", text="Select")
    tree.column("Select", width=50, anchor="center")
    tree.bind("<Button-1>", toggle_checkbox)  # Handle checkbox toggle
    tree.pack(fill="both", expand=True)
    filter_table("")

    # End Date Range
    ttk.Label(root, text="End Date:", font=("Arial", 12)).grid(row=3, column=0, padx=10, pady=10, sticky="e")
    end_date_entry = ttk.Entry(root, width=15)
    end_date_entry.grid(row=3, column=1, padx=10, pady=10, sticky="w")
    end_date_entry.insert(0, yesterday_str)

    # Buttons
    button_frame = ttk.Frame(root)
    button_frame.grid(row=4, column=0, columnspan=5, pady=10)

    run_button = ttk.Button(button_frame, text="Run Now", command=on_run_now)
    run_button.pack(side="left", padx=10)

    trade_table_button = ttk.Button(button_frame, text="Look at Trade Table", command=view_trade_table)
    trade_table_button.pack(side="left", padx=10)

    # New Export Button
    export_button = ttk.Button(button_frame, text="Export Results to CSV", command=export_results_to_csv)
    export_button.pack(side="left", padx=10)

    ttk.Label(root, text="Logs:", font=("Arial", 12)).grid(row=5, column=0, padx=10, pady=10, sticky="w")

    # Log Output
    log_text = tk.Text(root, height=15, width=120, wrap="word")  # Increased width for better visibility
    log_text.grid(row=6, column=0, columnspan=5, padx=10, pady=10)

    # Status Label
    status_label = ttk.Label(root, text="Ready", font=("Arial", 10))
    status_label.grid(row=7, column=0, columnspan=5, padx=10, pady=(0, 0), sticky='w')

    # Progress Bar
    progress_bar = ttk.Progressbar(root, mode="determinate", maximum=100)
    progress_bar.grid(row=8, column=0, columnspan=5, padx=10, pady=10, sticky="ew")

    # Initial UI State
    # **Remove conditional hiding of selection table**
    # tree_frame.grid_remove()
    # search_frame.grid_remove()

    root.mainloop()

if __name__ == "__main__":
    create_ui()
