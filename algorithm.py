# algorithm.py

import numpy as np
import pandas as pd
from datetime import datetime
import time

def run_algorithm(data, start_amount=10000, progress_callback=None, compounding=True):
    """
    Runs the SMA trading algorithm on the provided stock data.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Date', 'Close', and 'Ticker' columns.
        start_amount (float): Initial amount of liquidity.
        progress_callback (function): Function to call with progress updates (percentage).
        compounding (bool): If True, reinvest gains after each trade; if False, do not reinvest.
    
    Returns:
        dict: A dictionary containing output results and best trades.
    """
    # Ensure 'Close' is the stock price column
    stockcol = 'Close'  # Adjust if necessary based on your data

    # Convert 'Date' to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(data['Date']):
        data['Date'] = pd.to_datetime(data['Date'])

    # Round 'Close' prices to thousandths place
    data[stockcol] = data[stockcol].round(3)

    # Sort the data by 'Date' in ascending order
    stocks = data.sort_values('Date').reset_index(drop=True)

    # Optional prints for debugging
    print("Column Names:", stocks.columns.tolist())
    print("\nFirst 5 Dates After Sorting and Limiting:")
    print(stocks['Date'].head())
    print("\nLast 5 Dates After Sorting and Limiting:")
    print(stocks['Date'].tail())
    print("\nFirst 5 Rows After Parsing:")
    print(stocks.head())

    numrows = len(stocks)

    # Simple arrays for two SMAs
    sma1 = np.zeros(numrows)
    sma2 = np.zeros(numrows)

    # Tax rates (example)
    over1yeartax = 0.78
    under1yeartax = 0.65

    # Initialize best return variables
    besttaxedreturn = -np.inf  # Use negative infinity for initial comparison
    besta = None
    bestb = None
    besttradecount = 0
    besttrades = []
    bestendtaxed_liquidity = start_amount

    # For output stats (initialized)
    losingtrades = 0
    losingtradepct = 0
    winningtradepct = 0
    average_hold_time = 0
    win_percentage_last_4_trades = None

    # Parameters for SMA ranges
    astart, aend, bstart, bend, inc = 5, 200, 5, 200, 5
    combinations = (((aend - astart) // inc) + 1) * (((bend - bstart) // inc) + 1)
    iterations = 0

    # Loop through SMA combinations
    for a in range(astart, aend + 1, inc):
        # Recompute SMA1 for current 'a'
        sma1 = stocks[stockcol].rolling(window=a, min_periods=a).mean().fillna(0).values

        for b in range(bstart, bend + 1, inc):
            # Recompute SMA2 for current 'b'
            sma2 = stocks[stockcol].rolling(window=b, min_periods=b).mean().fillna(0).values
            smadiff = sma1 - sma2

            # Initialize buy/sell signals
            buysells = np.zeros(numrows)
            pos = 0  # 0: not in position, 1: in position

            # Depending on compounding, we handle liquidity differently
            if compounding:
                current_liquidity = start_amount
            else:
                # If not compounding, we’ll keep current_liquidity the same
                # but track profit separately in running_pnl
                current_liquidity = start_amount
                running_pnl = 0.0

            # We start evaluating signals at the index where SMA2 first becomes valid
            start_index = max(a, b) - 1

            # Generate buy/sell signals
            for i in range(start_index, numrows - 1):
                smadiff_current = smadiff[i]
                smadiff_prev = smadiff[i - 1]
                diff_change = smadiff_current - smadiff_prev

                if diff_change > 0 and pos == 0:   # indicates an upward crossover
                    buysells[i] = 1  # Buy
                    pos = 1
                elif diff_change < 0 and pos == 1: # indicates a downward crossover
                    buysells[i] = -1  # Sell
                    pos = 0
                else:
                    buysells[i] = 0   # No action

            # Calculate and list trades
            trades = []  # List to store trades as dictionaries
            tradecount = 0
            buy_index = None

            for i in range(start_index, numrows):
                signal = buysells[i]
                if signal == 1:  # Buy
                    tradecount += 1
                    trades.append({
                        'TradeNumber': tradecount,
                        'Buy/Sell': 1,
                        'DateNum': stocks.at[i, 'Date'].toordinal(),
                        'Price': stocks.at[i, stockcol],
                        'PreTaxReturn': 0.0,         # Will be updated on sell
                        'PreTaxCumReturn': 0.0,      # Will be updated
                        'HoldTime': 0.0,            # Will be updated on sell
                        'Date': stocks.at[i, 'Date'],
                        'PreTaxLiquidity': current_liquidity,
                        'PreTax Running P/L': 0.0    # Will be updated
                    })
                    buy_index = i

                elif signal == -1 and buy_index is not None:  # Sell
                    tradecount += 1
                    sell_price = stocks.at[i, stockcol]
                    buy_price = stocks.at[buy_index, stockcol]
                    pre_tax_return = (sell_price - buy_price) / buy_price
                    hold_time = (stocks.at[i, 'Date'] - stocks.at[buy_index, 'Date']).days

                    if compounding:
                        # Increase current liquidity (reinvest gains)
                        profit_dollars = current_liquidity * pre_tax_return
                        current_liquidity += profit_dollars
                    else:
                        # Non-compounding: Gains do not change the liquidity for the next trade,
                        # we just track profits separately
                        profit_dollars = start_amount * pre_tax_return
                        running_pnl += profit_dollars

                    trades.append({
                        'TradeNumber': tradecount,
                        'Buy/Sell': -1,
                        'DateNum': stocks.at[i, 'Date'].toordinal(),
                        'Price': sell_price,
                        'PreTaxReturn': pre_tax_return,
                        'PreTaxCumReturn': 0.0,  # Will be updated
                        'HoldTime': hold_time,
                        'Date': stocks.at[i, 'Date'],
                        'PreTaxLiquidity': current_liquidity
                            if compounding else (start_amount + running_pnl),
                        'PreTax Running P/L': profit_dollars
                    })
                    buy_index = None  # Reset after selling

            # Handle open position at the end by selling at the last price if not already sold
            if pos == 1 and buy_index is not None:
                if buysells[numrows - 1] != -1:
                    tradecount += 1
                    sell_price = stocks.at[numrows - 1, stockcol]
                    buy_price = stocks.at[buy_index, stockcol]
                    pre_tax_return = (sell_price - buy_price) / buy_price
                    hold_time = (stocks.at[numrows - 1, 'Date'] - stocks.at[buy_index, 'Date']).days

                    if compounding:
                        profit_dollars = current_liquidity * pre_tax_return
                        current_liquidity += profit_dollars
                    else:
                        profit_dollars = start_amount * pre_tax_return
                        running_pnl += profit_dollars

                    trades.append({
                        'TradeNumber': tradecount,
                        'Buy/Sell': -1,
                        'DateNum': stocks.at[numrows - 1, 'Date'].toordinal(),
                        'Price': sell_price,
                        'PreTaxReturn': pre_tax_return,
                        'PreTaxCumReturn': 0.0,
                        'HoldTime': hold_time,
                        'Date': stocks.at[numrows - 1, 'Date'],
                        'PreTaxLiquidity': current_liquidity
                            if compounding else (start_amount + running_pnl),
                        'PreTax Running P/L': profit_dollars
                    })

            # Convert trades list to a DataFrame
            trades_df = pd.DataFrame(trades)

            # Initialize cumulative variables
            pre_tax_pnl = 0.0
            pre_tax_cum_return = 0.0

            # Update trades with cumulative returns and P/L
            for idx_trade, row_trade in trades_df.iterrows():
                if row_trade['Buy/Sell'] == 1:
                    # Buy trade doesn't realize P/L yet
                    trades_df.at[idx_trade, 'PreTax Running P/L'] = pre_tax_pnl
                elif row_trade['Buy/Sell'] == -1:
                    # Sell trade: realize immediate P/L
                    trade_return = row_trade['PreTaxReturn']
                    # For the sake of clarity, if compounding:
                    # the final used liquidity was updated, so let's approximate
                    # the P/L from that trade. But in your original code, 
                    # you used `current_liquidity * pre_tax_return`.
                    # We can keep pre_tax_pnl as a sum of realized gains.
                    if compounding:
                        # Just assume the dictionary has the correct P/L in 'PreTax Running P/L'
                        pre_tax_pnl += row_trade['PreTax Running P/L']
                    else:
                        pre_tax_pnl += row_trade['PreTax Running P/L']

                    pre_tax_cum_return = (pre_tax_cum_return + 1) * (trade_return + 1) - 1
                    trades_df.at[idx_trade, 'PreTaxCumReturn'] = pre_tax_cum_return
                    trades_df.at[idx_trade, 'PreTax Running P/L'] = pre_tax_pnl

            # Convert back to list if needed
            if not trades_df.empty:
                trades = trades_df.to_dict('records')
            else:
                trades = []

            # Compute under1yearpl and over1yearpl
            # We use a simplified approach: if compounding is True, then at each sell, 
            # the "liquidity" changes. If not, each trade is from the original start_amount.
            under1yearpl = 0.0
            over1yearpl = 0.0

            if compounding:
                # We'll “simulate” the rolling liquidity for short vs. long-term splits
                rolling_liquidity = start_amount
                for tr in trades:
                    if tr['Buy/Sell'] == -1:
                        hold_time = tr['HoldTime']
                        # Gains from that trade in dollars:
                        gain_dollars = rolling_liquidity * tr['PreTaxReturn']
                        if hold_time < 365:
                            under1yearpl += gain_dollars
                        else:
                            over1yearpl += gain_dollars
                        rolling_liquidity += gain_dollars
            else:
                # Non-compounding: each trade is effectively from the same `start_amount`
                for tr in trades:
                    if tr['Buy/Sell'] == -1:
                        hold_time = tr['HoldTime']
                        gain_dollars = start_amount * tr['PreTaxReturn']
                        if hold_time < 365:
                            under1yearpl += gain_dollars
                        else:
                            over1yearpl += gain_dollars

            # Calculate taxed return
            tradecount_ = len([t for t in trades if t['Buy/Sell'] == -1])
            if tradecount_ > 0 and (under1yearpl + over1yearpl) > 0:
                taxed_under = under1yearpl * under1yeartax if under1yearpl > 0 else under1yearpl
                taxed_over = over1yearpl * over1yeartax if over1yearpl > 0 else over1yearpl
                taxcumpl = taxed_under + taxed_over
            else:
                taxcumpl = under1yearpl + over1yearpl

            # Compute final taxed liquidity and taxed return
            endtaxed_liquidity = start_amount + taxcumpl
            taxcumreturn = (endtaxed_liquidity / start_amount) - 1

            # Update best result if current is better
            if taxcumreturn > besttaxedreturn:
                besttaxedreturn = taxcumreturn
                besta = a
                bestb = b
                besttradecount = tradecount_
                besttrades = trades.copy()
                bestendtaxed_liquidity = endtaxed_liquidity

                # Collect stats
                sell_trades = [t for t in besttrades if t['Buy/Sell'] == -1]
                total_sell_trades = len(sell_trades)
                losingtrades = sum(1 for t in sell_trades if t.get('PreTaxReturn', 0) < 0)
                winningtrades = total_sell_trades - losingtrades

                losingtradepct = losingtrades / total_sell_trades if total_sell_trades else 0
                winningtradepct = winningtrades / total_sell_trades if total_sell_trades else 0

                hold_times = [t['HoldTime'] for t in sell_trades]
                average_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0

                last_4_trades = sell_trades[-4:]
                wins_last_4 = sum(1 for t in last_4_trades if t['PreTaxReturn'] > 0)
                win_percentage_last_4_trades = (wins_last_4 / 4) if len(last_4_trades) == 4 else None

            # Increment iteration, update progress if provided
            iterations += 1
            if progress_callback:
                progress_percentage = (iterations / combinations) * 100
                progress_percentage = min(progress_percentage, 100)  # ensure it does not exceed 100
                progress_callback(progress_percentage)

    # After exploring all (a, b) combos, calculate the "no algorithm" taxed return
    total_days = (stocks.at[numrows - 1, 'Date'] - stocks.at[0, 'Date']).days if numrows > 1 else 0
    price_return = (
        (stocks.at[numrows - 1, stockcol] - stocks.at[0, stockcol]) / stocks.at[0, stockcol]
    ) if numrows > 1 else 0

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

    # Compare final best taxed return vs. buy-and-hold taxed return
    if noalgoreturn != 0:
        betteroff = (besttaxedreturn - noalgoreturn) / abs(noalgoreturn)
    else:
        if besttaxedreturn > 0:
            betteroff = np.inf
        elif besttaxedreturn < 0:
            betteroff = -np.inf
        else:
            betteroff = 0

    # Average trade percentage: consider only sells
    sell_trades_final = [t for t in besttrades if t['Buy/Sell'] == -1]
    if sell_trades_final:
        avgtradepct = besttaxedreturn / len(sell_trades_final)
        maxdrawdown = min(t['PreTaxReturn'] for t in sell_trades_final)
    else:
        avgtradepct = 0
        maxdrawdown = 0

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
        "(noalgoreturn+1)*startamount": (noalgoreturn + 1) * start_amount,
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
        # Reorder columns for clarity
        desired_cols = [
            'TradeNumber','Buy/Sell','DateNum','Price','PreTaxReturn',
            'PreTaxCumReturn','HoldTime','Date','PreTaxLiquidity','PreTax Running P/L'
        ]
        # Filter only columns that exist (avoid errors if slightly different)
        existing_cols = [c for c in desired_cols if c in besttrades_df.columns]
        besttrades_df = besttrades_df[existing_cols]
    else:
        besttrades_df = pd.DataFrame()

    # Final result
    result = {
        "outputresults1": outputresults1,
        "outputresults2": outputresults2,
        "besttrades": besttrades_df.to_dict('records') if not besttrades_df.empty else []
    }

    return result
