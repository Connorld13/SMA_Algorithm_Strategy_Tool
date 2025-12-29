# algorithm.py

import numpy as np
import pandas as pd
from datetime import datetime
import time
import cache_manager

def run_algorithm(data, start_amount=10000, progress_callback=None, compounding=True, optimization_objective="taxed_return", 
                  start_date=None, end_date=None, use_cache=True, sma_a_start=5, sma_a_end=200, sma_b_start=5, sma_b_end=200, sma_inc=5):
    """
    Runs the SMA trading algorithm on the provided stock data.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Date', 'Close', and 'Ticker' columns.
        start_amount (float): Initial amount of liquidity.
        progress_callback (function): Function to call with progress updates (percentage).
        compounding (bool): If True, reinvest gains after each trade; if False, do not reinvest.
        optimization_objective (str): Objective to optimize for - "taxed_return", "better_off", or "win_rate"
        start_date (str or None): Start date for cache key generation
        end_date (str or None): End date for cache key generation
        use_cache (bool): Whether to use cached results if available
    
    Returns:
        dict: A dictionary containing output results, best trades, and all combinations.
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

    # Parameters for SMA ranges (can be overridden by function parameters)
    astart, aend, bstart, bend, inc = sma_a_start, sma_a_end, sma_b_start, sma_b_end, sma_inc
    
    # Validate ranges
    if astart < 1 or aend < astart or bstart < 1 or bend < bstart or inc < 1:
        return {"Error": f"Invalid SMA range parameters: SMA_A ({astart}-{aend}), SMA_B ({bstart}-{bend}), Increment ({inc})"}
    
    combinations = (((aend - astart) // inc) + 1) * (((bend - bstart) // inc) + 1)
    iterations = 0

    # Arrays to store all results for parameter stability analysis
    all_taxed_returns = []
    all_better_off = []
    all_win_rates = []
    all_trade_counts = []
    all_parameters = []  # Store (a, b) pairs
    
    # Store all combination results with detailed metrics for scoring
    all_combinations = []
    
    # Get ticker for cache key
    ticker = data['Ticker'].iloc[0] if 'Ticker' in data.columns and len(data) > 0 else "UNKNOWN"
    
    # Check cache if enabled
    use_cached_combinations = False
    if use_cache and start_date is not None and end_date is not None:
        cached_data = cache_manager.load_backtest_cache(
            ticker, start_date, end_date, compounding, optimization_objective, start_amount
        )
        if cached_data is not None and 'all_combinations' in cached_data:
            print(f"Loading cached combinations for {ticker} ({len(cached_data['all_combinations'])} combinations)")
            all_combinations = cached_data['all_combinations']
            use_cached_combinations = True
            
            # Extract data from cached combinations for compatibility
            all_taxed_returns = [c['taxed_return'] for c in all_combinations]
            all_better_off = [c.get('better_off', 0) for c in all_combinations]
            all_win_rates = [c['win_rate'] for c in all_combinations]
            all_trade_counts = [c['trade_count'] for c in all_combinations]
            all_parameters = [(c['sma_a'], c['sma_b']) for c in all_combinations]
            
            # Get noalgoreturn from cache or calculate it
            if 'noalgoreturn' in cached_data:
                noalgoreturn = cached_data['noalgoreturn']
            else:
                # Calculate noalgoreturn if not in cache
                total_days = (stocks.at[numrows - 1, 'Date'] - stocks.at[0, 'Date']).days if numrows > 1 else 0
                price_return = (
                    (stocks.at[numrows - 1, stockcol] - stocks.at[0, stockcol]) / stocks.at[0, stockcol]
                ) if numrows > 1 else 0
                if total_days < 365:
                    noalgoreturn = price_return * under1yeartax if price_return > 0 else price_return
                else:
                    noalgoreturn = price_return * over1yeartax if price_return > 0 else price_return

    # Loop through SMA combinations (skip if using cached data)
    if not use_cached_combinations:
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
                    # If not compounding, we'll keep current_liquidity the same
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
                    # We'll "simulate" the rolling liquidity for short vs. long-term splits
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

                # Calculate win rate for this combination
                sell_trades = [t for t in trades if t['Buy/Sell'] == -1]
                total_sell_trades = len(sell_trades)
                winning_trades = sum(1 for t in sell_trades if t.get('PreTaxReturn', 0) > 0)
                win_rate = winning_trades / total_sell_trades if total_sell_trades > 0 else 0

                # Calculate additional metrics for scoring
                losing_trades = total_sell_trades - winning_trades
                losing_trade_pct = losing_trades / total_sell_trades if total_sell_trades > 0 else 0
                
                # Calculate max drawdown (worst trade return)
                max_drawdown = min([t.get('PreTaxReturn', 0) for t in sell_trades]) if sell_trades else 0
                
                # Calculate average hold time
                hold_times = [t.get('HoldTime', 0) for t in sell_trades]
                avg_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0
                
                # Calculate average trade return
                avg_trade_return = sum([t.get('PreTaxReturn', 0) for t in sell_trades]) / len(sell_trades) if sell_trades else 0
                
                # Calculate return volatility (std dev of trade returns)
                trade_returns = [t.get('PreTaxReturn', 0) for t in sell_trades]
                return_std = np.std(trade_returns) if len(trade_returns) > 1 else 0
                
                # Calculate win percentage of last 4 trades
                last_4_trades = sell_trades[-4:] if len(sell_trades) >= 4 else sell_trades
                wins_last_4 = sum(1 for t in last_4_trades if t.get('PreTaxReturn', 0) > 0)
                win_pct_last_4 = (wins_last_4 / len(last_4_trades)) if len(last_4_trades) > 0 else None

                # Store results for parameter stability analysis
                all_taxed_returns.append(taxcumreturn)
                all_trade_counts.append(tradecount_)
                all_parameters.append((a, b))
                all_win_rates.append(win_rate)  # Store win rate here instead of recalculating later
                
                # Store detailed combination results for scoring
                # Convert Date objects to strings for JSON serialization in cache
                trades_for_cache = []
                for trade in trades:
                    trade_copy = trade.copy()
                    if 'Date' in trade_copy and hasattr(trade_copy['Date'], 'strftime'):
                        trade_copy['Date'] = trade_copy['Date'].strftime('%Y-%m-%d')
                    trades_for_cache.append(trade_copy)
                
                combination_result = {
                    'sma_a': a,
                    'sma_b': b,
                    'taxed_return': taxcumreturn,
                    'win_rate': win_rate,
                    'trade_count': tradecount_,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'losing_trade_pct': losing_trade_pct,
                    'max_drawdown': max_drawdown,
                    'avg_hold_time': avg_hold_time,
                    'avg_trade_return': avg_trade_return,
                    'return_std': return_std,
                    'win_pct_last_4': win_pct_last_4,
                    'end_taxed_liquidity': endtaxed_liquidity,
                    'under1yearpl': under1yearpl,
                    'over1yearpl': over1yearpl,
                    'trades': trades_for_cache  # Store trades for this combination
                }
                all_combinations.append(combination_result)

                # Increment iteration, update progress if provided
                iterations += 1
                if progress_callback:
                    progress_percentage = (iterations / combinations) * 100
                    progress_percentage = min(progress_percentage, 100)  # ensure it does not exceed 100
                    progress_callback(progress_percentage)

    # After exploring all (a, b) combos, calculate the "no algorithm" taxed return (if not cached)
    if not use_cached_combinations:
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

        # Calculate better off for all combinations and update all_combinations
        for i, tax_return in enumerate(all_taxed_returns):
            if noalgoreturn != 0:
                better_off = (tax_return - noalgoreturn) / abs(noalgoreturn)
            else:
                if tax_return > 0:
                    better_off = np.inf
                elif tax_return < 0:
                    better_off = -np.inf
                else:
                    better_off = 0
            all_better_off.append(better_off)
            # Update the combination result with better_off
            all_combinations[i]['better_off'] = better_off
    else:
        # Extract better_off from cached combinations if not present
        if not all_better_off or len(all_better_off) == 0:
            for i, c in enumerate(all_combinations):
                if 'better_off' not in c or c['better_off'] is None:
                    tax_return = c['taxed_return']
                    if noalgoreturn != 0:
                        better_off = (tax_return - noalgoreturn) / abs(noalgoreturn)
                    else:
                        if tax_return > 0:
                            better_off = np.inf
                        elif tax_return < 0:
                            better_off = -np.inf
                        else:
                            better_off = 0
                    all_combinations[i]['better_off'] = better_off
            all_better_off = [c.get('better_off', 0) for c in all_combinations]

    # Find best combination based on optimization objective
    if optimization_objective == "taxed_return":
        best_idx = np.argmax(all_taxed_returns)
        besttaxedreturn = all_taxed_returns[best_idx]
    elif optimization_objective == "better_off":
        best_idx = np.argmax(all_better_off)
        besttaxedreturn = all_taxed_returns[best_idx]
    elif optimization_objective == "win_rate":
        best_idx = np.argmax(all_win_rates)
        besttaxedreturn = all_taxed_returns[best_idx]
    else:
        # Default to taxed return
        best_idx = np.argmax(all_taxed_returns)
        besttaxedreturn = all_taxed_returns[best_idx]

    besta, bestb = all_parameters[best_idx]
    besttradecount = all_trade_counts[best_idx]

    # SECOND PASS: Calculate parameter stability metrics only around the optimal combination
    # Define the range around the optimal parameters (±10 SMA values)
    stability_range = 10
    a_stability_start = max(astart, besta - stability_range)
    a_stability_end = min(aend, besta + stability_range)
    b_stability_start = max(bstart, bestb - stability_range)
    b_stability_end = min(bend, bestb + stability_range)

    # Arrays to store stability analysis results
    stability_taxed_returns = []
    stability_better_off = []
    stability_win_rates = []
    stability_trade_counts = []
    
    # Calculate total combinations for stability analysis
    stability_combinations = (((a_stability_end - a_stability_start) // inc) + 1) * (((b_stability_end - b_stability_start) // inc) + 1)
    stability_iterations = 0

    # Loop through stability range combinations
    for a in range(a_stability_start, a_stability_end + 1, inc):
        # Recompute SMA1 for current 'a'
        sma1 = stocks[stockcol].rolling(window=a, min_periods=a).mean().fillna(0).values

        for b in range(b_stability_start, b_stability_end + 1, inc):
            # Recompute SMA2 for current 'b'
            sma2 = stocks[stockcol].rolling(window=b, min_periods=b).mean().fillna(0).values
            smadiff = sma1 - sma2

            # Initialize buy/sell signals
            buysells = np.zeros(numrows)
            pos = 0  # 0: not in position, 1: in position

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

            # Calculate trades for stability analysis (simplified version)
            trades_stability = []
            buy_index = None

            for i in range(start_index, numrows):
                signal = buysells[i]
                if signal == 1:  # Buy
                    buy_index = i
                elif signal == -1 and buy_index is not None:  # Sell
                    sell_price = stocks.at[i, stockcol]
                    buy_price = stocks.at[buy_index, stockcol]
                    pre_tax_return = (sell_price - buy_price) / buy_price
                    hold_time = (stocks.at[i, 'Date'] - stocks.at[buy_index, 'Date']).days

                    trades_stability.append({
                        'PreTaxReturn': pre_tax_return,
                        'HoldTime': hold_time
                    })
                    buy_index = None

            # Handle open position at the end
            if pos == 1 and buy_index is not None:
                sell_price = stocks.at[numrows - 1, stockcol]
                buy_price = stocks.at[buy_index, stockcol]
                pre_tax_return = (sell_price - buy_price) / buy_price
                hold_time = (stocks.at[numrows - 1, 'Date'] - stocks.at[buy_index, 'Date']).days

                trades_stability.append({
                    'PreTaxReturn': pre_tax_return,
                    'HoldTime': hold_time
                })

            # Calculate metrics for stability analysis
            tradecount_stability = len(trades_stability)
            
            # Calculate taxed return for stability analysis
            under1yearpl_stability = 0.0
            over1yearpl_stability = 0.0

            if compounding:
                rolling_liquidity = start_amount
                for tr in trades_stability:
                    hold_time = tr['HoldTime']
                    gain_dollars = rolling_liquidity * tr['PreTaxReturn']
                    if hold_time < 365:
                        under1yearpl_stability += gain_dollars
                    else:
                        over1yearpl_stability += gain_dollars
                    rolling_liquidity += gain_dollars
            else:
                for tr in trades_stability:
                    hold_time = tr['HoldTime']
                    gain_dollars = start_amount * tr['PreTaxReturn']
                    if hold_time < 365:
                        under1yearpl_stability += gain_dollars
                    else:
                        over1yearpl_stability += gain_dollars

            if tradecount_stability > 0 and (under1yearpl_stability + over1yearpl_stability) > 0:
                taxed_under = under1yearpl_stability * under1yeartax if under1yearpl_stability > 0 else under1yearpl_stability
                taxed_over = over1yearpl_stability * over1yeartax if over1yearpl_stability > 0 else over1yearpl_stability
                taxcumpl_stability = taxed_under + taxed_over
            else:
                taxcumpl_stability = under1yearpl_stability + over1yearpl_stability

            endtaxed_liquidity_stability = start_amount + taxcumpl_stability
            taxcumreturn_stability = (endtaxed_liquidity_stability / start_amount) - 1

            # Calculate win rate for stability analysis
            winning_trades_stability = sum(1 for t in trades_stability if t.get('PreTaxReturn', 0) > 0)
            win_rate_stability = winning_trades_stability / tradecount_stability if tradecount_stability > 0 else 0

            # Calculate better off for stability analysis
            if noalgoreturn != 0:
                better_off_stability = (taxcumreturn_stability - noalgoreturn) / abs(noalgoreturn)
            else:
                if taxcumreturn_stability > 0:
                    better_off_stability = np.inf
                elif taxcumreturn_stability < 0:
                    better_off_stability = -np.inf
                else:
                    better_off_stability = 0

            # Store stability results
            stability_taxed_returns.append(taxcumreturn_stability)
            stability_better_off.append(better_off_stability)
            stability_win_rates.append(win_rate_stability)
            stability_trade_counts.append(tradecount_stability)

            # Update progress for stability analysis
            stability_iterations += 1
            if progress_callback:
                # Combine progress from both passes
                first_pass_progress = 80  # First pass gets 80% of progress
                stability_progress = (stability_iterations / stability_combinations) * 20  # Second pass gets 20%
                total_progress = first_pass_progress + stability_progress
                progress_callback(total_progress)

    # Recalculate best trades for the best combination
    sma1 = stocks[stockcol].rolling(window=besta, min_periods=besta).mean().fillna(0).values
    sma2 = stocks[stockcol].rolling(window=bestb, min_periods=bestb).mean().fillna(0).values
    smadiff = sma1 - sma2

    # Generate best trades
    buysells = np.zeros(numrows)
    pos = 0
    start_index = max(besta, bestb) - 1

    for i in range(start_index, numrows - 1):
        smadiff_current = smadiff[i]
        smadiff_prev = smadiff[i - 1]
        diff_change = smadiff_current - smadiff_prev

        if diff_change > 0 and pos == 0:
            buysells[i] = 1
            pos = 1
        elif diff_change < 0 and pos == 1:
            buysells[i] = -1
            pos = 0
        else:
            buysells[i] = 0

    # Calculate best trades
    besttrades = []
    tradecount = 0
    buy_index = None
    current_liquidity = start_amount if compounding else start_amount
    running_pnl = 0.0

    for i in range(start_index, numrows):
        signal = buysells[i]
        if signal == 1:
            tradecount += 1
            besttrades.append({
                'TradeNumber': tradecount,
                'Buy/Sell': 1,
                'DateNum': stocks.at[i, 'Date'].toordinal(),
                'Price': stocks.at[i, stockcol],
                'PreTaxReturn': 0.0,
                'PreTaxCumReturn': 0.0,
                'HoldTime': 0.0,
                'Date': stocks.at[i, 'Date'],
                'PreTaxLiquidity': current_liquidity,
                'PreTax Running P/L': 0.0
            })
            buy_index = i

        elif signal == -1 and buy_index is not None:
            tradecount += 1
            sell_price = stocks.at[i, stockcol]
            buy_price = stocks.at[buy_index, stockcol]
            pre_tax_return = (sell_price - buy_price) / buy_price
            hold_time = (stocks.at[i, 'Date'] - stocks.at[buy_index, 'Date']).days

            if compounding:
                profit_dollars = current_liquidity * pre_tax_return
                current_liquidity += profit_dollars
            else:
                profit_dollars = start_amount * pre_tax_return
                running_pnl += profit_dollars

            besttrades.append({
                'TradeNumber': tradecount,
                'Buy/Sell': -1,
                'DateNum': stocks.at[i, 'Date'].toordinal(),
                'Price': sell_price,
                'PreTaxReturn': pre_tax_return,
                'PreTaxCumReturn': 0.0,  # Will be calculated below
                'HoldTime': hold_time,
                'Date': stocks.at[i, 'Date'],
                'PreTaxLiquidity': current_liquidity if compounding else (start_amount + running_pnl),
                'PreTax Running P/L': profit_dollars
            })
            buy_index = None

    # Handle open position at the end
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

            besttrades.append({
                'TradeNumber': tradecount,
                'Buy/Sell': -1,
                'DateNum': stocks.at[numrows - 1, 'Date'].toordinal(),
                'Price': sell_price,
                'PreTaxReturn': pre_tax_return,
                'PreTaxCumReturn': 0.0,  # Will be calculated below
                'HoldTime': hold_time,
                'Date': stocks.at[numrows - 1, 'Date'],
                'PreTaxLiquidity': current_liquidity if compounding else (start_amount + running_pnl),
                'PreTax Running P/L': profit_dollars
            })

    # Calculate PreTaxCumReturn for besttrades
    if besttrades:
        # Convert to DataFrame for easier manipulation
        besttrades_df = pd.DataFrame(besttrades)
        
        # Initialize cumulative return
        pre_tax_cum_return = 0.0
        
        # Update cumulative returns for sell trades
        for idx_trade, row_trade in besttrades_df.iterrows():
            if row_trade['Buy/Sell'] == -1:  # Sell trade
                trade_return = row_trade['PreTaxReturn']
                pre_tax_cum_return = (pre_tax_cum_return + 1) * (trade_return + 1) - 1
                besttrades_df.at[idx_trade, 'PreTaxCumReturn'] = pre_tax_cum_return
        
        # Convert back to list of dictionaries
        besttrades = besttrades_df.to_dict('records')

    # Calculate final metrics for best combination
    sell_trades_final = [t for t in besttrades if t['Buy/Sell'] == -1]
    total_sell_trades = len(sell_trades_final)
    losingtrades = sum(1 for t in sell_trades_final if t.get('PreTaxReturn', 0) < 0)
    winningtrades = total_sell_trades - losingtrades

    losingtradepct = losingtrades / total_sell_trades if total_sell_trades else 0
    winningtradepct = winningtrades / total_sell_trades if total_sell_trades else 0

    hold_times = [t['HoldTime'] for t in sell_trades_final]
    average_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0

    last_4_trades = sell_trades_final[-4:]
    wins_last_4 = sum(1 for t in last_4_trades if t['PreTaxReturn'] > 0)
    win_percentage_last_4_trades = (wins_last_4 / 4) if len(last_4_trades) == 4 else None

    # Calculate final taxed liquidity
    under1yearpl = 0.0
    over1yearpl = 0.0
    if compounding:
        rolling_liquidity = start_amount
        for tr in besttrades:
            if tr['Buy/Sell'] == -1:
                hold_time = tr['HoldTime']
                gain_dollars = rolling_liquidity * tr['PreTaxReturn']
                if hold_time < 365:
                    under1yearpl += gain_dollars
                else:
                    over1yearpl += gain_dollars
                rolling_liquidity += gain_dollars
    else:
        for tr in besttrades:
            if tr['Buy/Sell'] == -1:
                hold_time = tr['HoldTime']
                gain_dollars = start_amount * tr['PreTaxReturn']
                if hold_time < 365:
                    under1yearpl += gain_dollars
                else:
                    over1yearpl += gain_dollars

    if besttradecount > 0 and (under1yearpl + over1yearpl) > 0:
        taxed_under = under1yearpl * under1yeartax if under1yearpl > 0 else under1yearpl
        taxed_over = over1yearpl * over1yeartax if over1yearpl > 0 else over1yearpl
        taxcumpl = taxed_under + taxed_over
    else:
        taxcumpl = under1yearpl + over1yearpl

    bestendtaxed_liquidity = start_amount + taxcumpl

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
    if sell_trades_final:
        avgtradepct = besttaxedreturn / len(sell_trades_final)
        maxdrawdown = min(t['PreTaxReturn'] for t in sell_trades_final)
    else:
        avgtradepct = 0
        maxdrawdown = 0

    # Calculate parameter stability metrics using the stability analysis results
    # Convert to numpy arrays for easier calculations
    stability_taxed_returns = np.array(stability_taxed_returns)
    stability_better_off = np.array(stability_better_off)
    stability_win_rates = np.array(stability_win_rates)
    stability_trade_counts = np.array(stability_trade_counts)

    # Taxed Return stability metrics
    taxed_return_avg = np.mean(stability_taxed_returns)
    taxed_return_std = np.std(stability_taxed_returns)
    taxed_return_max = np.max(stability_taxed_returns)
    taxed_return_min = np.min(stability_taxed_returns)
    taxed_return_max_min_diff = taxed_return_max - taxed_return_min
    taxed_return_max_avg_diff = taxed_return_max - taxed_return_avg

    # Better Off stability metrics
    better_off_avg = np.mean(stability_better_off)
    better_off_std = np.std(stability_better_off)
    better_off_max = np.max(stability_better_off)
    better_off_min = np.min(stability_better_off)
    better_off_max_min_diff = better_off_max - better_off_min
    better_off_max_avg_diff = better_off_max - better_off_avg

    # Win Rate stability metrics
    win_rate_avg = np.mean(stability_win_rates)
    win_rate_std = np.std(stability_win_rates)
    win_rate_max = np.max(stability_win_rates)
    win_rate_min = np.min(stability_win_rates)
    win_rate_max_min_diff = win_rate_max - win_rate_min
    win_rate_max_avg_diff = win_rate_max - win_rate_avg

    # Trade Count stability metrics
    trade_count_avg = np.mean(stability_trade_counts)
    trade_count_std = np.std(stability_trade_counts)
    trade_count_max = np.max(stability_trade_counts)
    trade_count_min = np.min(stability_trade_counts)
    trade_count_max_min_diff = trade_count_max - trade_count_min
    trade_count_max_avg_diff = trade_count_max - trade_count_avg

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
        "combinations": combinations,
        "optimization_objective": optimization_objective,
        "stability_range": stability_range,
        "stability_combinations": stability_combinations
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

    # Parameter stability metrics
    param_stability = {
        # Taxed Return stability
        "taxed_return_avg": taxed_return_avg,
        "taxed_return_std": taxed_return_std,
        "taxed_return_max": taxed_return_max,
        "taxed_return_min": taxed_return_min,
        "taxed_return_max_min_diff": taxed_return_max_min_diff,
        "taxed_return_max_avg_diff": taxed_return_max_avg_diff,
        
        # Better Off stability
        "better_off_avg": better_off_avg,
        "better_off_std": better_off_std,
        "better_off_max": better_off_max,
        "better_off_min": better_off_min,
        "better_off_max_min_diff": better_off_max_min_diff,
        "better_off_max_avg_diff": better_off_max_avg_diff,
        
        # Win Rate stability
        "win_rate_avg": win_rate_avg,
        "win_rate_std": win_rate_std,
        "win_rate_max": win_rate_max,
        "win_rate_min": win_rate_min,
        "win_rate_max_min_diff": win_rate_max_min_diff,
        "win_rate_max_avg_diff": win_rate_max_avg_diff,
        
        # Trade Count stability
        "trade_count_avg": trade_count_avg,
        "trade_count_std": trade_count_std,
        "trade_count_max": trade_count_max,
        "trade_count_min": trade_count_min,
        "trade_count_max_min_diff": trade_count_max_min_diff,
        "trade_count_max_avg_diff": trade_count_max_avg_diff
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
        "param_stability": param_stability,
        "besttrades": besttrades_df.to_dict('records') if not besttrades_df.empty else [],
        "all_combinations": all_combinations,  # Store all combinations for rescoring
        "best_combination_idx": best_idx,  # Index of best combination
        "noalgoreturn": noalgoreturn
    }
    
    # Save to cache if enabled and we have date information
    # If start_date is None, use actual start date from data
    cache_start_date = start_date
    if use_cache and end_date is not None:
        if cache_start_date is None and numrows > 0:
            # Get actual start date from data
            actual_start = stocks['Date'].min()
            if hasattr(actual_start, 'strftime'):
                cache_start_date = actual_start.strftime("%Y-%m-%d")
            elif isinstance(actual_start, str):
                cache_start_date = actual_start
        
        if cache_start_date is not None:
            cache_manager.save_backtest_cache(
                ticker, cache_start_date, end_date, compounding, optimization_objective,
                start_amount, all_combinations, best_idx, noalgoreturn,
                besttrades=besttrades_df.to_dict('records') if not besttrades_df.empty else []
            )

    return result
