# algorithm.py

import pandas as pd
import numpy as np
from datetime import datetime

def run_matlab_sma_strategy(data, sma1_range=(5, 200), sma2_range=(5, 200), increment=5, start_amount=10000.0):
    """
    Run the SMA trading strategy converted from MATLAB code.

    Parameters:
        data (pd.DataFrame): DataFrame containing stock data with columns ['Date', 'Close', 'Ticker'].
        sma1_range (tuple): Tuple containing the start and end values for SMA1.
        sma2_range (tuple): Tuple containing the start and end values for SMA2.
        increment (int): Increment value for SMA ranges.
        start_amount (float): Initial amount to start trading with.

    Returns:
        dict: Results containing the best parameters and performance metrics for each ticker.
    """
    over1yeartax = 0.78
    under1yeartax = 0.65
    results = {}

    tickers = data["Ticker"].unique()
    for ticker in tickers:
        stock_data = data[data["Ticker"] == ticker].copy()
        stock_data.sort_values('Date', ascending=True, inplace=True)
        stock_data.reset_index(drop=True, inplace=True)

        numrows = len(stock_data)
        if numrows == 0:
            continue  # Skip if no data for ticker

        besttaxedreturn = -np.inf
        besta = None
        bestb = None
        besttradecount = 0
        besttrades = None
        bestunder1yearpl = 0
        bestover1yearpl = 0
        bestendtaxedliquidity = 0
        bestmaxdrawdown = None
        bestwinrate = None
        bestbetteroff = None
        bestnoalgoreturn = None
        test_date = stock_data.iloc[-1]['Date']

        astart, aend = sma1_range
        bstart, bend = sma2_range
        inc = increment

        combinations = (((aend - astart) // inc) + 1) * (((bend - bstart) // inc) + 1)
        iterations = 0

        # Adjusted loop order to match MATLAB (outer loop over SMA1)
        for a in range(astart, aend + 1, inc):
            # Compute SMA1 without shifting
            stock_data['SMA1'] = stock_data['Close'].rolling(window=a).mean()

            for b in range(bstart, bend + 1, inc):
                if a == b:
                    continue  # Skip if SMA1 period is equal to SMA2 period

                # Compute SMA2 without shifting
                stock_data['SMA2'] = stock_data['Close'].rolling(window=b).mean()

                # Skip if not enough data to compute SMAs
                if stock_data['SMA1'].isna().all() or stock_data['SMA2'].isna().all():
                    continue

                # Calculate SMADiff and its change
                stock_data['SMADiff'] = stock_data['SMA1'] - stock_data['SMA2']
                stock_data['SMADiff_Change'] = stock_data['SMADiff'] - stock_data['SMADiff'].shift(1)

                # Generate buy/sell signals
                stock_data['Position'] = 0
                pos = 0  # Position flag: 0 = no position, 1 = holding
                buysells = []
                # Start from index where both SMAs are valid
                start_idx = max(a, b)
                for idx in range(start_idx, numrows):
                    # Ensure SMADiff and SMADiff_Change are valid
                    if pd.isna(stock_data.at[idx, 'SMADiff']) or pd.isna(stock_data.at[idx - 1, 'SMADiff']):
                        continue
                    smadiff_change = stock_data.at[idx, 'SMADiff'] - stock_data.at[idx - 1, 'SMADiff']
                    if smadiff_change > 0 and pos == 0:
                        # Buy signal
                        stock_data.at[idx, 'Position'] = 1
                        buysells.append((idx, 1))
                        pos = 1
                    elif smadiff_change < 0 and pos == 1:
                        # Sell signal
                        stock_data.at[idx, 'Position'] = -1
                        buysells.append((idx, -1))
                        pos = 0
                    else:
                        stock_data.at[idx, 'Position'] = 0

                # Handle open position at the end of data
                if pos == 1:
                    idx = numrows - 1
                    stock_data.at[idx, 'Position'] = -1
                    buysells.append((idx, -1))
                    pos = 0

                # Simulate trades
                trades = []
                tradecount = 0
                for idx, signal in buysells:
                    tradecount += 1
                    trade = {
                        'TradeNumber': tradecount,
                        'Buy/Sell': signal,
                        'Date': stock_data.at[idx, 'Date'],
                        'Price': stock_data.at[idx, 'Close'],
                    }
                    trades.append(trade)

                if tradecount < 2:
                    continue  # Need at least one buy and one sell

                # Initialize trades_df with correct data types
                trades_df = pd.DataFrame(trades)

                # Explicitly set data types of columns
                trades_df = trades_df.astype({
                    'TradeNumber': 'int64',
                    'Buy/Sell': 'int64',
                    'Date': 'datetime64[ns]',
                    'Price': 'float64'
                })

                # Initialize other columns with correct data types
                trades_df['Return'] = pd.Series(0.0, index=trades_df.index, dtype='float64')
                trades_df['HoldTime'] = pd.Series(0, index=trades_df.index, dtype='int64')
                trades_df['CumulativeReturn'] = pd.Series(0.0, index=trades_df.index, dtype='float64')
                trades_df['Liquidity'] = pd.Series(np.nan, index=trades_df.index, dtype='float64')
                trades_df['P/L'] = pd.Series(0.0, index=trades_df.index, dtype='float64')

                # Set starting amount
                trades_df.at[0, 'Liquidity'] = float(start_amount)

                # Calculate returns and hold times
                for i in range(1, len(trades_df), 2):
                    buy_price = trades_df.at[i - 1, 'Price']
                    sell_price = trades_df.at[i, 'Price']
                    trades_df.at[i, 'Return'] = (sell_price - buy_price) / buy_price
                    buy_date = trades_df.at[i - 1, 'Date']
                    sell_date = trades_df.at[i, 'Date']
                    hold_time = (sell_date - buy_date).days
                    trades_df.at[i, 'HoldTime'] = int(hold_time)  # ensure integer

                    # Update cumulative return and liquidity
                    if i == 1:
                        trades_df.at[i, 'Liquidity'] = float(start_amount) * (1 + trades_df.at[i, 'Return'])
                    else:
                        prev_liquidity = trades_df.at[i - 2, 'Liquidity']
                        trades_df.at[i, 'Liquidity'] = prev_liquidity * (1 + trades_df.at[i, 'Return'])

                    trades_df.at[i, 'CumulativeReturn'] = (trades_df.at[i, 'Liquidity'] / float(start_amount)) - 1

                    # Ensure liquidity doesn't become negative
                    if trades_df.at[i, 'Liquidity'] <= 0:
                        print(f"Liquidity dropped to zero or negative at trade {i}.")
                        break

                # If no trades were executed properly, skip
                if len(trades_df) < 2:
                    continue

                # Calculate profit/loss for under and over 1-year holds
                under1yearpl = 0.0
                over1yearpl = 0.0
                for i in range(1, len(trades_df), 2):
                    if i == 1:
                        pl = trades_df.at[i, 'Liquidity'] - float(start_amount)
                    else:
                        pl = trades_df.at[i, 'Liquidity'] - trades_df.at[i - 2, 'Liquidity']
                    hold_time = trades_df.at[i, 'HoldTime']
                    if hold_time < 365:
                        under1yearpl += pl
                    else:
                        over1yearpl += pl

                # Calculate taxed cumulative return
                total_pl = under1yearpl + over1yearpl
                if total_pl > 0:
                    under1yearpl_calc = under1yearpl * under1yeartax if under1yearpl > 0 else under1yearpl
                    over1yearpl_calc = over1yearpl * over1yeartax if over1yearpl > 0 else over1yearpl
                    taxcumpl = under1yearpl_calc + over1yearpl_calc
                else:
                    taxcumpl = total_pl  # No tax if total profit is negative

                endtaxedliquidity = float(start_amount) + taxcumpl
                taxcumreturn = (endtaxedliquidity / float(start_amount)) - 1

                # Calculate buy-and-hold return without algorithm
                first_price = stock_data.iloc[0]['Close']
                last_price = stock_data.iloc[-1]['Close']
                hold_duration = (stock_data.iloc[-1]['Date'] - stock_data.iloc[0]['Date']).days
                raw_return = (last_price - first_price) / first_price

                if hold_duration < 365 and raw_return > 0:
                    noalgoreturn = raw_return * under1yeartax
                elif hold_duration >= 365 and raw_return > 0:
                    noalgoreturn = raw_return * over1yeartax
                else:
                    noalgoreturn = raw_return  # No tax if return is negative

                # Calculate better off metric
                if noalgoreturn < 0 and taxcumreturn >= 0:
                    betteroff = (taxcumreturn - noalgoreturn) / abs(noalgoreturn)
                elif noalgoreturn > 0:
                    betteroff = (taxcumreturn / noalgoreturn) - 1
                else:
                    betteroff = 0  # Both returns are negative or zero

                # Calculate losing trades and win rate
                losingtrades = 0
                total_closed_trades = tradecount // 2
                for i in range(1, len(trades_df), 2):
                    if trades_df.at[i, 'Return'] < 0:
                        losingtrades += 1
                win_rate = ((total_closed_trades - losingtrades) / total_closed_trades) if total_closed_trades > 0 else 0

                # Calculate max drawdown (maximum peak-to-trough decline)
                cumulative_liquidity = trades_df.loc[trades_df['Liquidity'].notnull(), 'Liquidity']
                peak = cumulative_liquidity.cummax()
                drawdown = (cumulative_liquidity - peak) / peak
                max_drawdown = drawdown.min()

                # Update best stats if current taxed return is better
                if taxcumreturn > besttaxedreturn:
                    besttaxedreturn = taxcumreturn
                    besta = a
                    bestb = b
                    besttradecount = tradecount
                    besttrades = trades_df.copy()
                    bestunder1yearpl = under1yearpl
                    bestover1yearpl = over1yearpl
                    bestendtaxedliquidity = endtaxedliquidity
                    bestbetteroff = betteroff
                    bestnoalgoreturn = noalgoreturn
                    bestmaxdrawdown = max_drawdown
                    bestwinrate = win_rate
                    bestlosingtrades = losingtrades  # Keep track of losing trades

                iterations += 1

        # If no trades were made, continue
        if besttrades is None:
            continue

        # Calculate losing trade percentage
        losingtradepct = (bestlosingtrades * 2) / besttradecount if besttradecount > 0 else 0

        # Calculate average trade percentage (avgtradepct)
        avgtradepct = besttaxedreturn / (besttradecount / 2) if besttradecount > 0 else 0

        # Store results
        results[ticker] = {
            'Ticker': ticker,
            'Test Date': test_date.strftime('%Y-%m-%d'),
            'Best SMA1': besta,
            'Best SMA2': bestb,
            '5 Year Diff %': bestbetteroff * 100,
            'Taxed Return %': besttaxedreturn * 100,
            'No Algo Return %': bestnoalgoreturn * 100,
            'Win Rate %': bestwinrate * 100,
            'Max Drawdown %': bestmaxdrawdown * 100,
            'Number of Closed Trades': besttradecount,
            'Average Trade %': avgtradepct * 100,
            'Losing Trade %': losingtradepct * 100,
            'Iterations': iterations,
            'Combinations': combinations,
        }

    return results
