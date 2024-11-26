import numpy as np
import pandas as pd
from datetime import datetime
import time

# Load and clean the data
data = pd.read_csv('data.csv')

# Flip the data to match MATLAB's flip function and reset the index
stocks = data.iloc[::-1].reset_index(drop=True)

# Print column names to verify correct mapping
print("Column Names:", stocks.columns.tolist())

# Convert 'Date' to datetime
stocks['Date'] = pd.to_datetime(stocks['Date'])

# Select and clean the stock price column
stockcol = 'Close/Last'  # Ensure this matches your data
if 'Close/Last' not in stocks.columns:
    stockcol = 'Close'  # Update accordingly

# Clean the 'Close/Last' column by removing '$' and stripping spaces
# Also, handle any potential commas in the numbers
stocks[stockcol] = (
    stocks[stockcol]
    .astype(str)  # Ensure all data is string type
    .str.replace('$', '', regex=False)  # Remove dollar signs
    .str.replace(',', '', regex=False)  # Remove commas if any
    .str.strip()  # Remove leading/trailing spaces
    .astype(float)  # Convert to float
)

# Verify parsing by printing first few rows
print("\nFirst 5 Rows After Parsing:")
print(stocks.head())

numrows = len(stocks)
startamount = 10000

# Initialize SMA1 and SMA2 arrays
sma1 = np.zeros(numrows)
sma2 = np.zeros(numrows)

# Tax rates
over1yeartax = 0.78
under1yeartax = 0.65

# Initialize best return variables
besttaxedreturn = -np.inf  # Use negative infinity for initial comparison
besta = None
bestb = None
besttradecount = 0
besttrades = []
bestendtaxed_liquidity = startamount

# Parameters for SMA ranges (set to 130 and 70)
astart, aend, bstart, bend, inc = 130, 130, 70, 70, 5
combinations = (((aend - astart) // inc) + 1) * (((bend - bstart) // inc) + 1)
iterations = 0

# Main optimization loop (only one iteration with a=130 and b=70)
for a in range(astart, aend + 1, inc):
    start_time_a = time.time()  # Start timing for 'a' loop (optional)

    # Compute SMA1 for current 'a'
    sma1 = stocks[stockcol].rolling(window=a, min_periods=a).mean().values

    for b in range(bstart, bend + 1, inc):
        # Compute SMA2 for current 'b'
        sma2 = stocks[stockcol].rolling(window=b, min_periods=b).mean().values

        # Initialize buy/sell signals
        buysells = np.zeros(numrows)
        pos = 0  # 0: not in position, 1: in position

        # Initialize current_liquidity before appending trades
        current_liquidity = startamount  # Initialize here to avoid NameError

        # Start index is b-1 to match MATLAB's 1-based indexing
        start_index = b-1

        # Generate buy/sell signals starting from the b-th index
        # Exclude the last day to prevent buy on the last day
        for i in range(start_index, numrows-1):
            # Only consider if both SMAs are valid
            if np.isnan(sma1[i]) or np.isnan(sma2[i]) or np.isnan(sma1[i - 1]) or np.isnan(sma2[i - 1]):
                continue  # Skip if any SMA value is NaN

            smadiff_current = sma1[i] - sma2[i]
            smadiff_prev = sma1[i - 1] - sma2[i - 1]
            diff_change = smadiff_current - smadiff_prev

            if diff_change > 0 and pos == 0:
                buysells[i] = 1  # Buy
                pos = 1
            elif diff_change < 0 and pos == 1:
                buysells[i] = -1  # Sell
                pos = 0
            else:
                buysells[i] = 0  # No action

        # Calculate and list trades
        trades = []  # List to store trades as dictionaries
        tradecount = 0
        buy_index = None

        for i in range(start_index, numrows):
            signal = buysells[i]
            if signal == 1:
                tradecount += 1
                trades.append({
                    'TradeNumber': tradecount,
                    'Buy/Sell': 1,
                    'DateNum': stocks.at[i, 'Date'].toordinal(),  # To match MATLAB's datenum
                    'Price': stocks.at[i, stockcol],
                    'PreTaxReturn': 0.0,  # Will be updated on sell
                    'PreTaxCumReturn': 0.0,  # Will be updated
                    'HoldTime': 0.0,  # Will be updated on sell
                    'Date': stocks.at[i, 'Date'],
                    'PreTaxLiquidity': startamount if tradecount == 1 else current_liquidity,
                    'PreTax Running P/L': 0.0  # Will be updated
                })
                buy_index = i
            elif signal == -1 and buy_index is not None:
                tradecount += 1
                sell_price = stocks.at[i, stockcol]
                buy_price = stocks.at[buy_index, stockcol]
                pre_tax_return = (sell_price - buy_price) / buy_price
                hold_time = (stocks.at[i, 'Date'] - stocks.at[buy_index, 'Date']).days

                trades.append({
                    'TradeNumber': tradecount,
                    'Buy/Sell': -1,
                    'DateNum': stocks.at[i, 'Date'].toordinal(),
                    'Price': sell_price,
                    'PreTaxReturn': pre_tax_return,
                    'PreTaxCumReturn': 0.0,  # Will be updated
                    'HoldTime': hold_time,
                    'Date': stocks.at[i, 'Date'],
                    'PreTaxLiquidity': current_liquidity + (current_liquidity * pre_tax_return),
                    'PreTax Running P/L': current_liquidity * pre_tax_return  # Will be updated
                })
                # Update current_liquidity after sell
                current_liquidity += current_liquidity * pre_tax_return
                buy_index = None  # Reset after selling

        # Handle open position at the end by selling at the last price if not already sold
        if pos == 1 and buy_index is not None:
            # Check if the last day already has a sell signal
            if buysells[numrows - 1] != -1:
                tradecount += 1
                sell_price = stocks.at[numrows - 1, stockcol]
                buy_price = stocks.at[buy_index, stockcol]
                pre_tax_return = (sell_price - buy_price) / buy_price
                hold_time = (stocks.at[numrows - 1, 'Date'] - stocks.at[buy_index, 'Date']).days

                trades.append({
                    'TradeNumber': tradecount,
                    'Buy/Sell': -1,
                    'DateNum': stocks.at[numrows - 1, 'Date'].toordinal(),
                    'Price': sell_price,
                    'PreTaxReturn': pre_tax_return,
                    'PreTaxCumReturn': 0.0,  # Will be updated later
                    'HoldTime': hold_time,
                    'Date': stocks.at[numrows - 1, 'Date'],
                    'PreTaxLiquidity': current_liquidity + (current_liquidity * pre_tax_return),
                    'PreTax Running P/L': current_liquidity * pre_tax_return  # Will be updated later
                })
                # Update current_liquidity after final sell
                current_liquidity += current_liquidity * pre_tax_return

        # Convert trades list to DataFrame
        trades_df = pd.DataFrame(trades)

        # Identification and verification of DateNum
        if not trades_df.empty:
            mismatches = trades_df.apply(
                lambda row: row['DateNum'] == row['Date'].toordinal(),
                axis=1
            )
            if not mismatches.all():
                mismatched_trades = trades_df[~mismatches]
                print("\nDateNum Mismatches Detected:")
                print(mismatched_trades)
            else:
                print("\nAll DateNum values correctly match their respective Dates.")
        else:
            print("\nNo trades generated for this combination.")

        # Initialize cumulative variables
        pre_tax_pnl = 0.0
        pre_tax_cum_return = 0.0

        # Update trades with cumulative returns and P/L
        for idx, trade in trades_df.iterrows():
            if trade['Buy/Sell'] == 1:
                # Buy Trade
                if trade['TradeNumber'] == 1:
                    trades_df.at[idx, 'PreTaxLiquidity'] = startamount
                else:
                    trades_df.at[idx, 'PreTaxLiquidity'] = current_liquidity
                trades_df.at[idx, 'PreTax Running P/L'] = pre_tax_pnl
            elif trade['Buy/Sell'] == -1:
                # Sell Trade
                pre_tax_return = trade['PreTaxReturn']
                pre_tax_pnl += current_liquidity * pre_tax_return
                pre_tax_cum_return = (pre_tax_cum_return + 1) * (pre_tax_return + 1) - 1
                trades_df.at[idx, 'PreTaxCumReturn'] = pre_tax_cum_return
                trades_df.at[idx, 'PreTax Running P/L'] = pre_tax_pnl

        # Assign the processed trades back to the list
        if not trades_df.empty:
            trades = trades_df.to_dict('records')
        else:
            trades = []

        # Now, compute under1yearpl and over1yearpl
        under1yearpl = 0.0
        over1yearpl = 0.0
        pre_tax_liquidity = startamount

        for trade in trades:
            if trade['Buy/Sell'] == -1:
                pre_tax_return = trade['PreTaxReturn']
                hold_time = trade['HoldTime']
                profit_dollars = pre_tax_return * pre_tax_liquidity

                if hold_time < 365:
                    under1yearpl += profit_dollars
                else:
                    over1yearpl += profit_dollars

                pre_tax_liquidity += profit_dollars  # Update liquidity after trade

        # Calculate taxed return
        if tradecount > 1 and (under1yearpl + over1yearpl) > 0:
            if under1yearpl > 0:
                taxed_under1yearpl = under1yearpl * under1yeartax
            else:
                taxed_under1yearpl = under1yearpl  # No tax on losses
            if over1yearpl > 0:
                taxed_over1yearpl = over1yearpl * over1yeartax
            else:
                taxed_over1yearpl = over1yearpl  # No tax on losses
            taxcumpl = taxed_under1yearpl + taxed_over1yearpl
        else:
            taxcumpl = under1yearpl + over1yearpl

        # Compute cumulative taxed liquidity and taxcumreturn
        endtaxed_liquidity = startamount + taxcumpl
        taxcumreturn = (endtaxed_liquidity / startamount) - 1

        # Update best return stats if current is better
        if taxcumreturn > besttaxedreturn:
            besttaxedreturn = taxcumreturn
            besta = a
            bestb = b
            besttradecount = tradecount
            besttrades = trades.copy()
            bestendtaxed_liquidity = endtaxed_liquidity

            # Calculate number of losing trades
            losingtrades = sum(1 for trade in besttrades if trade.get('PreTaxReturn', 0) < 0)
            losingtradepct = (losingtrades * 2) / besttradecount if besttradecount else 0  # Multiply by 2 to match MATLAB's logic

        iterations += 1

    # After all iterations, calculate overall stats
    total_days = (stocks.at[numrows - 1, 'Date'] - stocks.at[0, 'Date']).days
    price_return = (stocks.at[numrows - 1, stockcol] - stocks.at[0, stockcol]) / stocks.at[0, stockcol]

    # Calculate noalgoreturn based on holding period
    if total_days < 365:
        if price_return > 0:
            noalgoreturn = price_return * under1yeartax
        else:
            noalgoreturn = price_return
    else:
        if price_return > 0:
            noalgoreturn = price_return * over1yeartax
        else:
            noalgoreturn = price_return

    # Calculate betteroff
    if noalgoreturn != 0:
        betteroff = (besttaxedreturn / noalgoreturn) - 1
    else:
        betteroff = 0

    # Calculate average trade percentage: considering only sell trades
    sell_trades_final = [trade for trade in besttrades if trade['Buy/Sell'] == -1]
    avgtradepct = besttaxedreturn / len(sell_trades_final) if sell_trades_final else 0

    # Max drawdown: worst trade return percentage
    maxdrawdown = min(trade['PreTaxReturn'] for trade in sell_trades_final) if sell_trades_final else 0

    # Prepare output results
    outputresults1 = [
        betteroff,
        besttaxedreturn,
        noalgoreturn,
        besta,
        bestb,
        besttradecount,
        avgtradepct,
        iterations,
        combinations,
    ]

    outputresults2 = [
        startamount,
        bestendtaxed_liquidity,
        (noalgoreturn + 1) * startamount,
        losingtrades,
        losingtradepct,
        maxdrawdown,
    ]

    # Print results with appropriate formatting
    print("\nOutput Results 1:")
    print(f"betteroff={betteroff:.6f}")
    print(f"besttaxedreturn={besttaxedreturn:.6f}")
    print(f"noalgoreturn={noalgoreturn:.6f}")
    print(f"besta={besta}")
    print(f"bestb={bestb}")
    print(f"besttradecount={besttradecount}")
    print(f"avgtradepct={avgtradepct:.6f}")
    print(f"iterations={iterations}")
    print(f"combinations={combinations}")

    print("\nOutput Results 2:")
    print(f"startamount={startamount}")
    print(f"bestendtaxed_liquidity={bestendtaxed_liquidity:.2f}")
    print(f"(noalgoreturn+1)*startamount={ (noalgoreturn +1)*startamount:.2f}")
    print(f"losingtrades={losingtrades}")
    print(f"losingtradepct={losingtradepct:.5f}")
    print(f"maxdrawdown(worst trade return pct)={maxdrawdown:.6f}")

    # Set pandas display options to show all rows and full precision
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', lambda x: '%.12f' % x)

    # Convert besttrades to DataFrame for better visualization
    if besttrades:
        besttrades_df = pd.DataFrame(besttrades)

        # Assign column order
        besttrades_df = besttrades_df[['TradeNumber','Buy/Sell','DateNum','Price','PreTaxReturn','PreTaxCumReturn','HoldTime','Date','PreTaxLiquidity','PreTax Running P/L']]

        # Print the entire Best Trades DataFrame
        print("\nBest Trades:")
        print(besttrades_df)
    else:
        print("\nNo trades were generated.")