# algorithm.py

import numpy as np
import pandas as pd
from datetime import datetime
import time

def run_algorithm(data, start_amount=10000, progress_callback=None):
    """
    Runs the SMA trading algorithm on the provided stock data.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Date', 'Close', and 'Ticker' columns.
        start_amount (float): Initial amount of liquidity.
        progress_callback (function): Function to call with progress updates (percentage).

    Returns:
        dict: A dictionary containing output results and best trades.
    """
    # Ensure 'Close' is the stock price column
    stockcol = 'Close'  # Adjust if necessary based on your data

    # Convert 'Date' to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(data['Date']):
        data['Date'] = pd.to_datetime(data['Date'])

    # **Temporary Change: Round 'Close' prices to thousandths place**
    data[stockcol] = data[stockcol].round(3)

    # Sort the data by 'Date' in ascending order
    stocks = data.sort_values('Date').reset_index(drop=True)

    # Print column names to verify correct mapping
    print("Column Names:", stocks.columns.tolist())

    # Verify sorting by printing first and last few dates
    print("\nFirst 5 Dates After Sorting and Limiting:")
    print(stocks['Date'].head())
    print("\nLast 5 Dates After Sorting and Limiting:")
    print(stocks['Date'].tail())

    # Verify parsing by printing first few rows
    print("\nFirst 5 Rows After Parsing:")
    print(stocks.head())

    numrows = len(stocks)

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
    bestendtaxed_liquidity = start_amount

    # Parameters for SMA ranges
    astart, aend, bstart, bend, inc = 5, 200, 5, 200, 5
    combinations = (((aend - astart) // inc) + 1) * (((bend - bstart) // inc) + 1)
    iterations = 0

    # Loop through SMA combinations
    for a in range(astart, aend + 1, inc):
        # Recompute SMA1 for current 'a'
        sma1 = stocks[stockcol].rolling(window=a, min_periods=a).mean().fillna(0).values
        smadiff = sma1 - sma2  # Initial smadiff based on SMA1 and SMA2=0

        for b in range(bstart, bend + 1, inc):
            # Recompute SMA2 for current 'b'
            sma2 = stocks[stockcol].rolling(window=b, min_periods=b).mean().fillna(0).values
            smadiff = sma1 - sma2  # Recalculate smadiff based on updated SMA2

            # Initialize buy/sell signals
            buysells = np.zeros(numrows)
            pos = 0  # 0: not in position, 1: in position

            # Initialize current_liquidity before appending trades
            current_liquidity = start_amount

            # Set start_index to b - 1
            start_index = b - 1  # Start at index (b - 1)

            # Signal generation
            for i in range(start_index, numrows - 1):
                if i == start_index:
                    # Handle the first day where SMA1 may not be defined
                    smadiff_current = smadiff[i]
                    smadiff_prev = smadiff[i - 1]

                    diff_change = smadiff_current - smadiff_prev

                    if diff_change > 0 and pos == 0:
                        buysells[i] = 1  # Buy
                        pos = 1
                    elif diff_change < 0 and pos == 1:
                        buysells[i] = -1  # Sell
                        pos = 0
                    else:
                        buysells[i] = 0  # No action
                else:
                    # Existing logic for subsequent days
                    smadiff_current = smadiff[i]
                    smadiff_prev = smadiff[i - 1]
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
                        'DateNum': stocks.at[i, 'Date'].toordinal(),
                        'Price': stocks.at[i, stockcol],
                        'PreTaxReturn': 0.0,  # Will be updated on sell
                        'PreTaxCumReturn': 0.0,  # Will be updated
                        'HoldTime': 0.0,  # Will be updated on sell
                        'Date': stocks.at[i, 'Date'],
                        'PreTaxLiquidity': start_amount if tradecount == 1 else current_liquidity,
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

            # Initialize cumulative variables
            pre_tax_pnl = 0.0
            pre_tax_cum_return = 0.0

            # Update trades with cumulative returns and P/L
            for idx_trade, trade in trades_df.iterrows():
                if trade['Buy/Sell'] == 1:
                    # Buy Trade
                    if trade['TradeNumber'] == 1:
                        trades_df.at[idx_trade, 'PreTaxLiquidity'] = start_amount
                    else:
                        trades_df.at[idx_trade, 'PreTaxLiquidity'] = current_liquidity
                    trades_df.at[idx_trade, 'PreTax Running P/L'] = pre_tax_pnl
                elif trade['Buy/Sell'] == -1:
                    # Sell Trade
                    pre_tax_return = trade['PreTaxReturn']
                    pre_tax_pnl += current_liquidity * pre_tax_return
                    pre_tax_cum_return = (pre_tax_cum_return + 1) * (pre_tax_return + 1) - 1
                    trades_df.at[idx_trade, 'PreTaxCumReturn'] = pre_tax_cum_return
                    trades_df.at[idx_trade, 'PreTax Running P/L'] = pre_tax_pnl

            # Assign the processed trades back to the list
            if not trades_df.empty:
                trades = trades_df.to_dict('records')
            else:
                trades = []

            # Compute under1yearpl and over1yearpl
            under1yearpl = 0.0
            over1yearpl = 0.0
            pre_tax_liquidity = start_amount

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
            endtaxed_liquidity = start_amount + taxcumpl
            taxcumreturn = (endtaxed_liquidity / start_amount) - 1

            # Update best return stats if current is better
            if taxcumreturn > besttaxedreturn:
                besttaxedreturn = taxcumreturn
                besta = a
                bestb = b
                besttradecount = tradecount
                besttrades = trades.copy()
                bestunder1yearpl = under1yearpl
                bestover1yearpl = over1yearpl
                bestendtaxed_liquidity = endtaxed_liquidity

                # Calculate number of losing trades
                sell_trades = [trade for trade in besttrades if trade['Buy/Sell'] == -1]
                total_sell_trades = len(sell_trades)
                losingtrades = sum(1 for trade in sell_trades if trade.get('PreTaxReturn', 0) < 0)
                winningtrades = total_sell_trades - losingtrades

                # Calculate percentages
                losingtradepct = losingtrades / total_sell_trades if total_sell_trades else 0
                winningtradepct = winningtrades / total_sell_trades if total_sell_trades else 0

                # Calculate average hold time
                hold_times = [trade['HoldTime'] for trade in sell_trades]
                average_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0

                # Calculate win percentage of last 4 trades
                last_4_trades = sell_trades[-4:]
                wins_last_4 = sum(1 for trade in last_4_trades if trade['PreTaxReturn'] > 0)
                win_percentage_last_4_trades = wins_last_4 / 4 if len(last_4_trades) == 4 else None  # None if less than 4 trades

            iterations += 1

            # **Call progress callback with percentage progress**
            if progress_callback:
                progress_percentage = (iterations / combinations) * 100
                progress_percentage = min(progress_percentage, 100)  # Ensure it does not exceed 100
                progress_callback(progress_percentage)

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

    # Calculate betteroff using the new formula
    if noalgoreturn != 0:
        betteroff = (besttaxedreturn - noalgoreturn) / abs(noalgoreturn)
    else:
        # Handle the case where noalgoreturn is zero
        betteroff = np.inf if besttaxedreturn > 0 else -np.inf if besttaxedreturn < 0 else 0

    # Calculate average trade percentage: considering only sell trades
    sell_trades_final = [trade for trade in besttrades if trade['Buy/Sell'] == -1]
    avgtradepct = besttaxedreturn / len(sell_trades_final) if sell_trades_final else 0

    # Max drawdown: worst trade return percentage
    maxdrawdown = min(trade['PreTaxReturn'] for trade in sell_trades_final) if sell_trades_final else 0

    # Prepare output results
    outputresults1 = {
        "betteroff": betteroff,
        "besttaxedreturn": besttaxedreturn,
        "noalgoreturn": noalgoreturn,
        "besta": besta,
        "bestb": bestb,
        "besttradecount": besttradecount,
        "avgtradepct": avgtradepct,
        "iterations": iterations,
        "combinations": combinations
    }

    outputresults2 = {
        "startamount": start_amount,
        "bestendtaxed_liquidity": bestendtaxed_liquidity,
        "(noalgoreturn+1)*startamount": (noalgoreturn +1)*start_amount,
        "losingtrades": losingtrades,
        "losingtradepct": losingtradepct,
        "winningtradepct": winningtradepct,
        "maxdrawdown(worst trade return pct)": maxdrawdown,
        "average_hold_time": average_hold_time,
        "win_percentage_last_4_trades": win_percentage_last_4_trades
    }

    # Convert besttrades to DataFrame for better visualization
    if besttrades:
        besttrades_df = pd.DataFrame(besttrades)
        # Assign column order
        besttrades_df = besttrades_df[['TradeNumber','Buy/Sell','DateNum','Price','PreTaxReturn','PreTaxCumReturn','HoldTime','Date','PreTaxLiquidity','PreTax Running P/L']]
    else:
        besttrades_df = pd.DataFrame()

    # Prepare the final result dictionary
    result = {
        "outputresults1": outputresults1,
        "outputresults2": outputresults2,
        "besttrades": besttrades_df.to_dict('records') if not besttrades_df.empty else []
    }

    return result
