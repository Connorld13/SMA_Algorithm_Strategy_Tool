import yfinance as yf
import backtrader as bt
import pandas as pd
from datetime import datetime, timedelta

class SMADiffStrategy(bt.Strategy):
    params = (
        ('sma1', 10),  # Period for the first SMA
        ('sma2', 20),  # Period for the second SMA
        ('over1yeartax', 0.78),  # Tax rate for holding > 1 year
        ('under1yeartax', 0.65),  # Tax rate for holding <= 1 year
    )

    def __init__(self):
        # Calculate the SMAs
        self.sma1 = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma1)
        self.sma2 = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma2)

        # Difference between the two SMAs
        self.diff = self.sma1 - self.sma2

        # Initialize previous difference
        self.diff_prev = self.diff(-1)  # Placeholder for previous difference

        # Store trades for the trade table
        self.trade_log = []
        self.entry_price = None
        self.entry_date = None

    def notify_trade(self, trade):
        if trade.isclosed:
            exit_date = self.data.datetime.date(0)
            exit_price = trade.price
            pnl = trade.pnl

            # Calculate holding period
            if self.entry_date:
                holding_period = (exit_date - self.entry_date).days
            else:
                holding_period = 0

            # Determine tax rate based on holding period
            tax_rate = self.params.over1yeartax if holding_period > 365 else self.params.under1yeartax
            taxed_pnl = pnl * tax_rate if pnl > 0 else pnl

            self.trade_log.append(
                {
                    'Entry Date': self.entry_date.strftime('%Y-%m-%d') if self.entry_date else None,
                    'Exit Date': exit_date.strftime('%Y-%m-%d'),
                    'Holding Period': holding_period,
                    'Entry Price': self.entry_price if self.entry_price else None,
                    'Exit Price': exit_price,
                    'Profit/Loss': pnl,
                    'Taxed P/L': taxed_pnl,
                }
            )

            # Reset entry variables
            self.entry_price = None
            self.entry_date = None

    def next(self):
        # Update previous difference
        if len(self.diff) > 1:
            self.diff_prev = self.diff[-1]
        else:
            self.diff_prev = 0

        current_diff = self.diff[0]
        previous_diff = self.diff_prev

        # Trade logic: Only consider end-of-day prices and SMA difference
        if current_diff > previous_diff and not self.position:
            # Buy if the difference is increasing and no current position
            size = self.broker.getcash() / self.data.close[0]
            self.buy(size=size)
            self.entry_price = self.data.close[0]
            self.entry_date = self.data.datetime.date(0)

        elif current_diff < previous_diff and self.position:
            # Sell if the difference is decreasing and holding a position
            self.close()

def fetch_stock_data(ticker, start_date, end_date):
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)

    # Ensure the data is a pandas DataFrame and not empty
    if data is None or data.empty:
        raise ValueError(f"No data fetched for {ticker}. Check the ticker symbol or date range.")

    # Rename columns to match Backtrader's expected format
    data = data.rename(
        columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
    )

    # Add the 'openinterest' column (required by Backtrader)
    data['openinterest'] = 0

    # Ensure the DataFrame index is properly formatted as a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("The data index is not a DatetimeIndex. Ensure the fetched data has a valid datetime format.")

    # Check for required columns
    required_columns = {'open', 'high', 'low', 'close', 'volume', 'openinterest'}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"The data is missing one or more required columns: {required_columns}")

    # Debug print to check the structure of the DataFrame
    print(data.head())

    return data

def run_backtest(ticker, sma1, sma2, start_date, end_date, starting_cash=10000, over1yeartax=0.78, under1yeartax=0.65):
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(starting_cash)

    # Fetch and add data feed
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    data = bt.feeds.PandasData(dataname=stock_data)
    cerebro.adddata(data)

    # Add strategy
    cerebro.addstrategy(
        SMADiffStrategy, sma1=sma1, sma2=sma2, over1yeartax=over1yeartax, under1yeartax=under1yeartax
    )

    # Run backtest
    results = cerebro.run()
    strat = results[0]

    # Output trade table
    if strat.trade_log:
        print("\nTrade Table:")
        print("{:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
            "Entry Date", "Exit Date", "Holding (days)", "Entry Price", "Exit Price", "P/L", "Taxed P/L"
        ))
        for trade in strat.trade_log:
            print("{:<15} {:<15} {:<15} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f}".format(
                trade['Entry Date'],
                trade['Exit Date'],
                trade['Holding Period'],
                trade['Entry Price'],
                trade['Exit Price'],
                trade['Profit/Loss'],
                trade['Taxed P/L']
            ))
    else:
        print("\nNo trades were made.")

    # Print cumulative taxed return
    final_value = cerebro.broker.getvalue()
    cumulative_return = (final_value - starting_cash) / starting_cash * 100
    print(f"\nFinal Portfolio Value: ${final_value:.2f}")
    print(f"Cumulative Return (after taxes): {cumulative_return:.2f}%")

    # Plot the strategy
    cerebro.plot()

if __name__ == "__main__":
    try:
        # Input stock symbol and SMA periods
        ticker = 'AAPL'
        sma1 = 82
        sma2 = 67

        # Calculate start_date (5 years before yesterday) and end_date (yesterday)
        end_date_dt = datetime.now() - timedelta(days=1)
        end_date = end_date_dt.strftime('%Y-%m-%d')
        start_date_dt = end_date_dt - timedelta(days=5 * 365)
        start_date = start_date_dt.strftime('%Y-%m-%d')

        print(f"\nRunning backtest for {ticker} with SMA1={sma1} and SMA2={sma2}")
        print(f"Data range: {start_date} to {end_date}")

        # Run the backtest
        run_backtest(ticker, sma1, sma2, start_date, end_date)
    except Exception as e:
        print(f"Error: {e}")
