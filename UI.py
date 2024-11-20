import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
from fetchdata import fetch_and_prepare_data
from algorithm import run_matlab_sma_strategy  # Corrected import

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
    if mode in ["10 Stocks", "Single Stock"]:
        tree_frame.grid()
        search_frame.grid()
    else:
        tree_frame.grid_remove()
        search_frame.grid_remove()

def toggle_checkbox(event):
    """Toggle the checkbox state in the Treeview."""
    mode = mode_var.get()
    region = tree.identify_region(event.x, event.y)
    if region == "cell":
        row_id = tree.identify_row(event.y)
        col_id = tree.identify_column(event.x)
        if col_id == "#3":  # Checkbox column
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
    return selected_stocks

def filter_table(query):
    """Filter the table based on the search query."""
    for row in tree.get_children():
        tree.delete(row)
    for _, row in sp500_data.iterrows():
        if query.lower() in row["Symbol"].lower() or query.lower() in row["Name"].lower():
            tree.insert("", "end", values=(row["Symbol"], row["Name"], ""), tags=("row",))

def on_run_now():
    """Fetch data and execute the strategy."""
    global log_text  # Ensure log_text is recognized globally

    try:
        mode = mode_var.get()

        # Get date inputs
        start_date = start_date_entry.get()
        end_date = end_date_entry.get()

        # Validate dates
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")

        # Determine selected stocks based on mode
        if mode == "Entire S&P 500":
            selected_stocks = sp500_data['Symbol'].tolist()
        elif mode == "Half of S&P 500":
            selected_stocks = sp500_data['Symbol'].tolist()[:len(sp500_data)//2]
        else:
            # Validate stock selection
            selected_stocks = validate_stock_selection()
            if selected_stocks is None:
                return

        # DEBUG: Log stock selection and dates
        print(f"Selected stocks: {selected_stocks}")
        print(f"Date range: {start_date} to {end_date}")

        # Fetch data
        print("Fetching data...")
        data = fetch_and_prepare_data(selected_stocks, start_date, end_date)

        # DEBUG: Log fetched data
        print(f"Fetched data:\n{data.head()}")

        # Run the SMA strategy
        print("Running SMA strategy...")
        sma1_range = (5, 200)
        sma2_range = (5, 200)
        increment = 5
        results = run_matlab_sma_strategy(data, sma1_range, sma2_range, increment)

        # DEBUG: Log strategy results
        print(f"Strategy results:\n{results}")

        # Display results in log text
        log_text.delete(1.0, tk.END)  # Clear previous logs
        for ticker, result in results.items():
            log_text.insert(tk.END, f"Results for {ticker}:\n")
            for key, value in result.items():
                log_text.insert(tk.END, f"  {key}: {value}\n")
            log_text.insert(tk.END, "\n")

        # Removed CSV export code as per your request

    except ValueError as e:
        messagebox.showerror("Error", str(e))
        print(f"ValueError: {e}")
    except Exception as e:
        messagebox.showerror("Error", f"Unexpected error: {e}")
        print(f"Unexpected error: {e}")

def create_ui():
    # Default date range: 5 years from yesterday
    end_date_default = datetime.now() - timedelta(days=1)
    start_date_default = end_date_default - relativedelta(years=5)
    end_date_str = end_date_default.strftime("%Y-%m-%d")
    start_date_str = start_date_default.strftime("%Y-%m-%d")

    global root, mode_var, sp500_data, tree, search_entry, tree_frame, search_frame
    global start_date_entry, end_date_entry, log_text  # Declare log_text as global here

    # Load S&P 500 tickers
    sp500_data = fetch_sp500_tickers_from_csv()

    # Main Window
    root = tk.Tk()
    root.title("SMA Trading Simulation")
    root.geometry("900x700")

    # Mode Selection
    mode_var = tk.StringVar(value="Single Stock")
    ttk.Label(root, text="Select Mode:", font=("Arial", 12)).grid(row=0, column=0, padx=10, pady=10, sticky="w")
    modes = ["Entire S&P 500", "Half of S&P 500", "10 Stocks", "Single Stock"]
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
    tree = ttk.Treeview(tree_frame, columns=("Ticker", "Name", "Select"), show="headings", yscrollcommand=tree_scroll.set)
    tree_scroll.config(command=tree.yview)
    tree.heading("Ticker", text="Ticker")
    tree.heading("Name", text="Name")
    tree.heading("Select", text="Select")
    tree.column("Select", width=50, anchor="center")
    tree.bind("<Button-1>", toggle_checkbox)  # Handle checkbox toggle
    tree.pack(fill="both", expand=True)
    filter_table("")

    # Date Range
    ttk.Label(root, text="Date Range:", font=("Arial", 12)).grid(row=3, column=0, padx=10, pady=10, sticky="w")
    ttk.Label(root, text="Start Date:").grid(row=3, column=1, padx=10, pady=5, sticky="e")
    start_date_entry = ttk.Entry(root, width=15)
    start_date_entry.grid(row=3, column=2, padx=10, pady=5, sticky="w")
    start_date_entry.insert(0, start_date_str)

    ttk.Label(root, text="End Date:").grid(row=3, column=3, padx=10, pady=5, sticky="e")
    end_date_entry = ttk.Entry(root, width=15)
    end_date_entry.grid(row=3, column=4, padx=10, pady=5, sticky="w")
    end_date_entry.insert(0, end_date_str)

    # Buttons
    ttk.Button(root, text="Run Now", command=on_run_now).grid(row=4, column=1, padx=10, pady=20)
    ttk.Label(root, text="Logs:", font=("Arial", 12)).grid(row=5, column=0, padx=10, pady=10, sticky="w")

    # Log Output
    log_text = tk.Text(root, height=10, width=100)  # Create log_text here
    log_text.grid(row=6, column=0, columnspan=5, padx=10, pady=10)

    # Initial UI State
    tree_frame.grid_remove()
    search_frame.grid_remove()
    root.mainloop()

if __name__ == "__main__":
    create_ui()
