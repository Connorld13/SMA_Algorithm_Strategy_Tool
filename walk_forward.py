# walk_forward.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import algorithm
import scoring

def run_walk_forward_analysis(data, start_amount=10000, progress_callback=None, compounding=True,
                              optimization_objective="taxed_return", end_date=None,
                              backtest_period_years=4, backtest_period_months=0,
                              walk_forward_period_years=1, walk_forward_period_months=0,
                              rebalance_years=0, rebalance_months=3, rebalance_none=False,
                              scoring_config=None):
    """
    Run walk-forward analysis on stock data.
    
    Parameters:
        data: DataFrame with Date, Close, Ticker columns
        start_amount: Starting capital
        progress_callback: Function to call with progress updates
        compounding: Whether to compound gains
        optimization_objective: Objective to optimize ("taxed_return", "better_off", "win_rate")
        end_date: End date for analysis (YYYY-MM-DD format)
        backtest_period_years/months: Training period length
        walk_forward_period_years/months: Testing period length
        rebalance_years/months: How often to re-optimize (0,0 means only initial optimization)
        rebalance_none: If True, only optimize once at the start
        scoring_config: Scoring configuration for final ranking
    
    Returns:
        Dictionary with aggregated walk-forward results
    """
    if scoring_config is None:
        scoring_config = scoring.get_default_scoring_config()
    
    # Convert end_date to datetime
    if isinstance(end_date, str):
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    else:
        end_date_dt = end_date
    
    # Ensure data is sorted by date
    if not pd.api.types.is_datetime64_any_dtype(data['Date']):
        data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date').reset_index(drop=True)
    
    # Filter data up to end_date
    data = data[data['Date'] <= end_date_dt].copy()
    
    if len(data) == 0:
        raise ValueError("No data available for walk-forward analysis")
    
    # Walk-forward: Calculate periods backward from end_date
    # Test period is the walk-forward period ending at end_date
    test_end = end_date_dt
    test_period = relativedelta(years=walk_forward_period_years, months=walk_forward_period_months)
    test_start = test_end - test_period + timedelta(days=1)  # Start test period (walk_forward_period before end_date)
    
    # Training period is the backtest_period ending just before test period
    train_end = test_start - timedelta(days=1)  # Training ends the day before test starts
    train_period = relativedelta(years=backtest_period_years, months=backtest_period_months)
    train_start = train_end - train_period + timedelta(days=1)  # Training starts (backtest_period before train_end)
    
    # Ensure we have enough data
    data_min = data['Date'].min()
    data_max = data['Date'].max()
    
    if train_start < data_min:
        raise ValueError(f"Not enough data for walk-forward. Training period requires data from {train_start.date()}, but earliest data is {data_min.date()}.")
    
    if test_end > data_max:
        raise ValueError(f"Not enough data for walk-forward. Test period requires data until {test_end.date()}, but latest data is {data_max.date()}.")
    
    if progress_callback:
        progress_callback(10)
    
    # Debug logging for time periods
    print(f"\n[WALK-FORWARD TIME DEBUG]")
    print(f"  Input end_date: {end_date_dt.date()}")
    print(f"  Data date range: {data['Date'].min().date()} to {data['Date'].max().date()}")
    print(f"  Training period input: {backtest_period_years} years, {backtest_period_months} months")
    print(f"  Walk-forward period input: {walk_forward_period_years} years, {walk_forward_period_months} months")
    print(f"  Calculated Training Period: {train_start.date()} to {train_end.date()}")
    print(f"  Calculated Test Period: {test_start.date()} to {test_end.date()}")
    print(f"  Training data rows: {len(data[(data['Date'] >= train_start) & (data['Date'] <= train_end)])}")
    print(f"  Test data rows: {len(data[(data['Date'] >= test_start) & (data['Date'] <= test_end)])}")
    print(f"  Overlap check: train_end ({train_end.date()}) < test_start ({test_start.date()}): {train_end < test_start}")
    print(f"  Total period: {(test_end - train_start).days / 365.25:.2f} years")
    
    # Get training data
    train_data = data[(data['Date'] >= train_start) & (data['Date'] <= train_end)].copy()
    
    if len(train_data) == 0:
        raise ValueError("No training data available")
    
    # Run optimization on training period
    if progress_callback:
        progress_callback(20)
    
    train_result = algorithm.run_algorithm(
        train_data,
        start_amount=start_amount,
        progress_callback=lambda p: progress_callback(20 + p * 0.4) if progress_callback else None,
        compounding=compounding,
        optimization_objective=optimization_objective,
        start_date=train_start.strftime("%Y-%m-%d"),
        end_date=train_end.strftime("%Y-%m-%d"),
        use_cache=True
    )
    
    if "Error" in train_result:
        return train_result
    
    # Get best SMA pair from training (using scoring to find best)
    all_combinations_train = train_result.get('all_combinations', [])
    if not all_combinations_train:
        return {"Error": "No combinations found in training period"}
    
    # Score all training combinations and find best
    best_train_combo = None
    best_train_score = -np.inf
    best_train_idx = 0
    
    for idx, combo in enumerate(all_combinations_train):
        combo_result = {
            "outputresults1": {
                "besttaxedreturn": combo.get("taxed_return", 0),
                "betteroff": combo.get("better_off", 0),
                "besttradecount": combo.get("trade_count", 0),
                "noalgoreturn": train_result.get("noalgoreturn", 0)
            },
            "outputresults2": {
                "winningtradepct": combo.get("win_rate", 0),
                "maxdrawdown(worst trade return pct)": combo.get("max_drawdown", 0),
                "average_hold_time": combo.get("avg_hold_time", 0)
            },
            "param_stability": train_result.get("param_stability", {})
        }
        score = scoring.calculate_backtest_score(combo_result, scoring_config)
        if score > best_train_score:
            best_train_score = score
            best_train_combo = combo
            best_train_idx = idx
    
    if best_train_combo is None:
        return {"Error": "Could not determine best training combination"}
    
    best_a = best_train_combo['sma_a']
    best_b = best_train_combo['sma_b']
    
    # Get training trades for best combination
    training_trades = train_result.get('besttrades', [])
    # Add metadata to training trades
    for trade in training_trades:
        if isinstance(trade.get('Date'), datetime):
            trade_date = trade['Date']
        else:
            trade_date = pd.to_datetime(trade.get('Date', train_start))
        trade['Period'] = 'Training'
        trade['SMA_A'] = best_a
        trade['SMA_B'] = best_b
    
    # Calculate training score
    training_score = best_train_score
    
    # Get test data
    test_data = data[(data['Date'] >= test_start) & (data['Date'] <= test_end)].copy()
    
    if len(test_data) == 0:
        return {"Error": "No test data available"}
    
    if progress_callback:
        progress_callback(65)
    
    # Test the best pair on test period
    test_result = run_fixed_parameters_backtest(
        test_data,
        sma_a=best_a,
        sma_b=best_b,
        start_amount=start_amount,
        compounding=compounding,
        start_date=test_start.strftime("%Y-%m-%d"),
        end_date=test_end.strftime("%Y-%m-%d")
    )
    
    if test_result is None:
        return {"Error": "Failed to run test period backtest"}
    
    if progress_callback:
        progress_callback(85)
    
    # Get walk-forward trades
    walk_forward_trades = test_result.get('trades', [])
    
    # Add metadata to walk-forward trades
    for trade in walk_forward_trades:
        trade['Period'] = 'Walk-Forward Test'
        trade['SMA_A'] = best_a
        trade['SMA_B'] = best_b
    
    # Calculate walk-forward score
    test_result_formatted = {
        "outputresults1": {
            "besttaxedreturn": test_result.get('return', 0.0),
            "betteroff": 0.0,
            "besttradecount": test_result.get('trade_count', 0),
            "noalgoreturn": 0.0
        },
        "outputresults2": {
            "winningtradepct": test_result.get('winning_trades', 0) / max(test_result.get('trade_count', 1), 1),
            "maxdrawdown(worst trade return pct)": test_result.get('max_drawdown', 0.0),
            "average_hold_time": test_result.get('total_hold_time', 0.0) / max(test_result.get('trade_count', 1), 1)
        },
        "param_stability": {
            "taxed_return_std": 0.0,
            "better_off_std": 0.0,
            "win_rate_std": 0.0,
            "taxed_return_max_min_diff": 0.0
        }
    }
    
    walk_forward_score = scoring.calculate_backtest_score(test_result_formatted, scoring_config)
    
    # Create all_combinations list (just the best one for now, but formatted for compatibility)
    all_combinations = [{
        'sma_a': best_a,
        'sma_b': best_b,
        'taxed_return': test_result.get('return', 0.0),
        'better_off': 0.0,
        'win_rate': test_result.get('winning_trades', 0) / max(test_result.get('trade_count', 1), 1),
        'trade_count': test_result.get('trade_count', 0),
        'winning_trades': test_result.get('winning_trades', 0),
        'losing_trades': test_result.get('losing_trades', 0),
        'max_drawdown': test_result.get('max_drawdown', 0.0),
        'avg_hold_time': test_result.get('total_hold_time', 0.0) / max(test_result.get('trade_count', 1), 1),
        'combined_score': walk_forward_score
    }]
    
    # Find best combination (only one in simple mode)
    best_combo = all_combinations[0] if all_combinations else None
    best_idx = 0
    
    # Format output similar to regular algorithm
    outputresults1 = {
        "betteroff": 0.0,
        "besttaxedreturn": test_result.get('return', 0.0),
        "noalgoreturn": 0.0,
        "besta": best_a,
        "bestb": best_b,
        "besttradecount": test_result.get('trade_count', 0),
        "avgtradepct": 0.0,
        "iterations": 0,
        "combinations": len(all_combinations),
        "optimization_objective": optimization_objective,
        "stability_range": 0,
        "stability_combinations": 0
    }
    
    outputresults2 = {
        "startamount": start_amount,
        "bestendtaxed_liquidity": start_amount * (1 + test_result.get('return', 0.0)),
        "(noalgoreturn+1)*startamount": start_amount,
        "losingtrades": test_result.get('losing_trades', 0),
        "losingtradepct": 1 - (test_result.get('winning_trades', 0) / max(test_result.get('trade_count', 1), 1)),
        "winningtradepct": test_result.get('winning_trades', 0) / max(test_result.get('trade_count', 1), 1),
        "maxdrawdown(worst trade return pct)": test_result.get('max_drawdown', 0.0),
        "average_hold_time": test_result.get('total_hold_time', 0.0) / max(test_result.get('trade_count', 1), 1),
        "win_percentage_last_4_trades": None
    }
    
    param_stability = train_result.get("param_stability", {})
    
    # Calculate training and walk-forward metrics
    training_metrics = {
        "taxed_return": best_train_combo.get('taxed_return', 0.0),
        "better_off": best_train_combo.get('better_off', 0.0),
        "trade_count": best_train_combo.get('trade_count', 0),
        "win_rate": best_train_combo.get('win_rate', 0.0),
        "max_drawdown": best_train_combo.get('max_drawdown', 0.0),
        "avg_hold_time": best_train_combo.get('avg_hold_time', 0.0),
        "winning_trades": best_train_combo.get('winning_trades', 0),
        "losing_trades": best_train_combo.get('losing_trades', 0)
    }
    
    walk_forward_metrics = {
        "taxed_return": test_result.get('return', 0.0),
        "better_off": 0.0,
        "trade_count": test_result.get('trade_count', 0),
        "win_rate": test_result.get('winning_trades', 0) / max(test_result.get('trade_count', 1), 1),
        "max_drawdown": test_result.get('max_drawdown', 0.0),
        "avg_hold_time": test_result.get('total_hold_time', 0.0) / max(test_result.get('trade_count', 1), 1),
        "winning_trades": test_result.get('winning_trades', 0),
        "losing_trades": test_result.get('losing_trades', 0)
    }
    
    return {
        "outputresults1": outputresults1,
        "outputresults2": outputresults2,
        "param_stability": param_stability,
        "besttrades": training_trades,  # Training period trades
        "all_combinations": all_combinations,
        "best_combination_idx": 0,
        "noalgoreturn": train_result.get("noalgoreturn", 0.0),
        "walk_forward_mode": True,
        "segments": 1,  # Simple mode has 1 segment
        "training_score": training_score,
        "walk_forward_score": walk_forward_score,
        "combined_score": (training_score * scoring_config.get("combined_score_weighting", {}).get("training_weight", 0.4) + 
                          walk_forward_score * scoring_config.get("combined_score_weighting", {}).get("walk_forward_weight", 0.6)),
        "training_trades": training_trades,  # Store training trades separately
        "walk_forward_trades": walk_forward_trades,  # Store walk-forward trades separately
        "training_period": {"start": train_start, "end": train_end},
        "test_period": {"start": test_start, "end": test_end},
        "best_sma_a": best_a,
        "best_sma_b": best_b,
        "training_metrics": training_metrics,
        "walk_forward_metrics": walk_forward_metrics
    }


def run_fixed_parameters_backtest(data, sma_a, sma_b, start_amount=10000, compounding=True,
                                  start_date=None, end_date=None):
    """
    Run backtest with fixed SMA parameters (no optimization).
    Used for testing optimized parameters on walk-forward periods.
    """
    try:
        # This is a simplified version - we'll reuse algorithm logic
        # For now, call the main algorithm but with a very narrow parameter range
        # that forces it to use the specified values
        
        # Actually, we need to implement a simpler version that just runs the strategy
        # with fixed parameters. Let me create a minimal implementation.
        
        stockcol = 'Close'
        if not pd.api.types.is_datetime64_any_dtype(data['Date']):
            data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date').reset_index(drop=True)
        data[stockcol] = data[stockcol].round(3)
        
        numrows = len(data)
        if numrows < max(sma_a, sma_b):
            return None
        
        # Calculate SMAs
        sma1 = data[stockcol].rolling(window=sma_a, min_periods=sma_a).mean().fillna(0).values
        sma2 = data[stockcol].rolling(window=sma_b, min_periods=sma_b).mean().fillna(0).values
        smadiff = sma1 - sma2
        
        # Generate signals
        buysells = np.zeros(numrows)
        pos = 0
        start_index = max(sma_a, sma_b) - 1
        
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
        
        # Calculate trades
        trades = []
        buy_index = None
        current_liquidity = start_amount
        
        for i in range(start_index, numrows):
            signal = buysells[i]
            if signal == 1:
                buy_index = i
            elif signal == -1 and buy_index is not None:
                sell_price = data.at[i, stockcol]
                buy_price = data.at[buy_index, stockcol]
                buy_date = data.at[buy_index, 'Date']
                sell_date = data.at[i, 'Date']
                pre_tax_return = (sell_price - buy_price) / buy_price
                hold_time = (sell_date - buy_date).days
                
                if compounding:
                    gain_dollars = current_liquidity * pre_tax_return
                    current_liquidity += gain_dollars
                else:
                    gain_dollars = start_amount * pre_tax_return
                
                trades.append({
                    'Date': sell_date,  # Use sell date as trade date
                    'BuyDate': buy_date,
                    'SellDate': sell_date,
                    'BuyPrice': buy_price,
                    'SellPrice': sell_price,
                    'PreTaxReturn': pre_tax_return,
                    'HoldTime': hold_time,
                    'GainDollars': gain_dollars
                })
                buy_index = None
        
        # Handle open position
        if pos == 1 and buy_index is not None:
            sell_price = data.at[numrows - 1, stockcol]
            buy_price = data.at[buy_index, stockcol]
            buy_date = data.at[buy_index, 'Date']
            sell_date = data.at[numrows - 1, 'Date']
            pre_tax_return = (sell_price - buy_price) / buy_price
            hold_time = (sell_date - buy_date).days
            
            if compounding:
                gain_dollars = current_liquidity * pre_tax_return
            else:
                gain_dollars = start_amount * pre_tax_return
            
            trades.append({
                'Date': sell_date,  # Use sell date as trade date
                'BuyDate': buy_date,
                'SellDate': sell_date,
                'BuyPrice': buy_price,
                'SellPrice': sell_price,
                'PreTaxReturn': pre_tax_return,
                'HoldTime': hold_time,
                'GainDollars': gain_dollars
            })
        
        # Calculate metrics
        if len(trades) == 0:
            return {
                'trade_count': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'return': 0.0,
                'pnl': 0.0,
                'max_drawdown': 0.0,
                'total_hold_time': 0.0
            }
        
        # Apply taxes
        over1yeartax = 0.78
        under1yeartax = 0.65
        
        total_pnl = 0.0
        for trade in trades:
            hold_time = trade['HoldTime']
            gain = trade['GainDollars']
            if hold_time < 365:
                total_pnl += gain * under1yeartax if gain > 0 else gain
            else:
                total_pnl += gain * over1yeartax if gain > 0 else gain
        
        final_liquidity = start_amount + total_pnl
        total_return = (final_liquidity / start_amount) - 1.0
        
        winning_trades = sum(1 for t in trades if t['PreTaxReturn'] > 0)
        losing_trades = len(trades) - winning_trades
        max_drawdown = min([t['PreTaxReturn'] for t in trades]) if trades else 0.0
        total_hold_time = sum([t['HoldTime'] for t in trades])
        
        return {
            'trade_count': len(trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'return': total_return,
            'pnl': total_pnl,
            'max_drawdown': max_drawdown,
            'total_hold_time': total_hold_time,
            'trades': trades  # Include the trades list
        }
        
    except Exception as e:
        print(f"Error in fixed parameters backtest: {e}")
        import traceback
        traceback.print_exc()
        return None

import algorithm
import scoring

def run_batch_walk_forward_analysis(data, start_amount=10000, progress_callback=None, compounding=True,
                                   optimization_objective="taxed_return", end_date=None,
                                   backtest_period_years=4, backtest_period_months=0,
                                   walk_forward_period_years=1, walk_forward_period_months=0,
                                   scoring_config=None, training_result=None):
    """
    Run simple walk-forward analysis for batch mode: Train on first X years, test on remaining years.
    This is optimized for batch runs where we first run regular algorithm to find best combo.
    """
    
    if scoring_config is None:
        scoring_config = scoring.get_default_scoring_config()
    
    # Convert end_date to datetime
    if isinstance(end_date, str):
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    else:
        end_date_dt = end_date
    
    # Ensure data is sorted by date
    if not pd.api.types.is_datetime64_any_dtype(data['Date']):
        data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date').reset_index(drop=True)
    
    # Filter data up to end_date
    data = data[data['Date'] <= end_date_dt].copy()
    
    if len(data) == 0:
        return {"Error": "No data available for walk-forward analysis"}
    
    # Walk-forward: Calculate periods backward from end_date
    # Test period is the walk-forward period ending at end_date
    test_end = end_date_dt
    test_period = relativedelta(years=walk_forward_period_years, months=walk_forward_period_months)
    test_start = test_end - test_period + timedelta(days=1)  # Start test period (walk_forward_period before end_date)
    
    # Training period is the backtest_period ending just before test period
    train_end = test_start - timedelta(days=1)  # Training ends the day before test starts
    train_period = relativedelta(years=backtest_period_years, months=backtest_period_months)
    train_start = train_end - train_period + timedelta(days=1)  # Training starts (backtest_period before train_end)
    
    # Ensure we have enough data
    data_min = data['Date'].min()
    data_max = data['Date'].max()
    
    if train_start < data_min:
        return {"Error": f"Not enough data for walk-forward. Training period requires data from {train_start.date()}, but earliest data is {data_min.date()}."}
    
    if test_end > data_max:
        return {"Error": f"Not enough data for walk-forward. Test period requires data until {test_end.date()}, but latest data is {data_max.date()}."}
    
    if progress_callback:
        progress_callback(10)
    
    # Debug logging for time periods
    print(f"\n[BATCH WALK-FORWARD TIME DEBUG]")
    print(f"  Input end_date: {end_date_dt.date()}")
    print(f"  Data date range: {data['Date'].min().date()} to {data['Date'].max().date()}")
    print(f"  Training period input: {backtest_period_years} years, {backtest_period_months} months")
    print(f"  Walk-forward period input: {walk_forward_period_years} years, {walk_forward_period_months} months")
    print(f"  Calculated Training Period: {train_start.date()} to {train_end.date()}")
    print(f"  Calculated Test Period: {test_start.date()} to {test_end.date()}")
    print(f"  Training data rows: {len(data[(data['Date'] >= train_start) & (data['Date'] <= train_end)])}")
    print(f"  Test data rows: {len(data[(data['Date'] >= test_start) & (data['Date'] <= test_end)])}")
    print(f"  Overlap check: train_end ({train_end.date()}) < test_start ({test_start.date()}): {train_end < test_start}")
    print(f"  Total period: {(test_end - train_start).days / 365.25:.2f} years")
    print(f"  Training result provided: {training_result is not None}")
    print(f"  ALWAYS recalculating training on training period only (ignoring provided training_result)")
    
    # Always recalculate training on training period only (ignore provided training_result from full timeframe)
    # Get training data
    train_data = data[(data['Date'] >= train_start) & (data['Date'] <= train_end)].copy()
    
    if len(train_data) == 0:
        return {"Error": "No training data available"}
    
    print(f"  Running algorithm on training period only: {train_start.date()} to {train_end.date()}")
    
    if progress_callback:
        progress_callback(20)
    
    # Always run training on training period only
    training_result = algorithm.run_algorithm(
        train_data,
        start_amount=start_amount,
        progress_callback=lambda p: progress_callback(20 + p * 0.4) if progress_callback else None,
        compounding=compounding,
        optimization_objective=optimization_objective,
        start_date=train_start.strftime("%Y-%m-%d"),
        end_date=train_end.strftime("%Y-%m-%d"),
        use_cache=True
    )
    
    print(f"  Training algorithm completed. Training result date range check:")
    if "Error" not in training_result:
        training_trades = training_result.get('besttrades', [])
        if training_trades:
            training_trade_dates = [pd.to_datetime(t.get('Date', train_start)) for t in training_trades if 'Date' in t]
            if training_trade_dates:
                print(f"    Training trades date range: {min(training_trade_dates).date()} to {max(training_trade_dates).date()}")
                print(f"    Expected training range: {train_start.date()} to {train_end.date()}")
                print(f"    Training trades within bounds: {min(training_trade_dates) >= train_start and max(training_trade_dates) <= train_end}")
    
    if "Error" in training_result:
        return training_result
    
    # Get best combo from training (using scoring config to find best scored combo)
    all_combinations = training_result.get('all_combinations', [])
    if not all_combinations:
        return {"Error": "No combinations found in training result"}
    
    # Score all combinations and find best
    best_combo = None
    best_score = -float('inf')
    best_idx = 0
    
    for idx, combo in enumerate(all_combinations):
        combo_result = {
            "outputresults1": {
                "besttaxedreturn": combo.get("taxed_return", 0),
                "betteroff": combo.get("better_off", 0),
                "besttradecount": combo.get("trade_count", 0),
                "noalgoreturn": training_result.get("noalgoreturn", 0)
            },
            "outputresults2": {
                "winningtradepct": combo.get("win_rate", 0),
                "maxdrawdown(worst trade return pct)": combo.get("max_drawdown", 0),
                "average_hold_time": combo.get("avg_hold_time", 0)
            },
            "param_stability": training_result.get("param_stability", {})
        }
        score = scoring.calculate_backtest_score(combo_result, scoring_config)
        if score > best_score:
            best_score = score
            best_combo = combo
            best_idx = idx
    
    if best_combo is None:
        return {"Error": "Could not determine best combination"}
    
    best_a = best_combo['sma_a']
    best_b = best_combo['sma_b']
    
    # Get training trades for best combination
    training_trades = training_result.get('besttrades', [])
    # Add metadata to training trades
    for trade in training_trades:
        if isinstance(trade.get('Date'), datetime):
            trade_date = trade['Date']
        else:
            trade_date = pd.to_datetime(trade.get('Date', train_start))
        trade['Period'] = 'Training'
        trade['SMA_A'] = best_a
        trade['SMA_B'] = best_b
    
    # Calculate training score
    training_score = best_score
    
    # Get test data
    test_data = data[(data['Date'] >= test_start) & (data['Date'] <= test_end)].copy()
    
    if len(test_data) == 0:
        return {"Error": "No test data available"}
    
    if progress_callback:
        progress_callback(65)
    
    # Test the best pair on test period
    test_result = run_fixed_parameters_backtest(
        test_data,
        sma_a=best_a,
        sma_b=best_b,
        start_amount=start_amount,
        compounding=compounding,
        start_date=test_start.strftime("%Y-%m-%d"),
        end_date=test_end.strftime("%Y-%m-%d")
    )
    
    if test_result is None:
        return {"Error": "Failed to run test period backtest"}
    
    if progress_callback:
        progress_callback(85)
    
    # Get walk-forward trades
    walk_forward_trades = test_result.get('trades', [])
    
    # Add metadata to walk-forward trades
    for trade in walk_forward_trades:
        trade['Period'] = 'Walk-Forward Test'
        trade['SMA_A'] = best_a
        trade['SMA_B'] = best_b
    
    # Calculate walk-forward score
    test_result_formatted = {
        "outputresults1": {
            "besttaxedreturn": test_result.get('return', 0.0),
            "betteroff": 0.0,
            "besttradecount": test_result.get('trade_count', 0),
            "noalgoreturn": 0.0
        },
        "outputresults2": {
            "winningtradepct": test_result.get('winning_trades', 0) / max(test_result.get('trade_count', 1), 1),
            "maxdrawdown(worst trade return pct)": test_result.get('max_drawdown', 0.0),
            "average_hold_time": test_result.get('total_hold_time', 0.0) / max(test_result.get('trade_count', 1), 1)
        },
        "param_stability": {
            "taxed_return_std": 0.0,
            "better_off_std": 0.0,
            "win_rate_std": 0.0,
            "taxed_return_max_min_diff": 0.0
        }
    }
    
    walk_forward_score = scoring.calculate_backtest_score(test_result_formatted, scoring_config)
    
    # Format results similar to regular walk forward
    outputresults1 = {
        "besta": best_a,
        "bestb": best_b,
        "besttaxedreturn": test_result.get('return', 0.0),
        "betteroff": 0.0,
        "besttradecount": test_result.get('trade_count', 0),
        "noalgoreturn": 0.0,
        "optimization_objective": optimization_objective
    }
    
    outputresults2 = {
        "winningtradepct": test_result.get('winning_trades', 0) / max(test_result.get('trade_count', 1), 1),
        "maxdrawdown(worst trade return pct)": test_result.get('max_drawdown', 0.0),
        "average_hold_time": test_result.get('total_hold_time', 0.0) / max(test_result.get('trade_count', 1), 1)
    }
    
    # Calculate training and walk-forward metrics
    training_metrics = {
        "taxed_return": best_combo.get('taxed_return', 0.0),
        "better_off": best_combo.get('better_off', 0.0),
        "trade_count": best_combo.get('trade_count', 0),
        "win_rate": best_combo.get('win_rate', 0.0),
        "max_drawdown": best_combo.get('max_drawdown', 0.0),
        "avg_hold_time": best_combo.get('avg_hold_time', 0.0),
        "winning_trades": best_combo.get('winning_trades', 0),
        "losing_trades": best_combo.get('losing_trades', 0)
    }
    
    walk_forward_metrics = {
        "taxed_return": test_result.get('return', 0.0),
        "better_off": 0.0,
        "trade_count": test_result.get('trade_count', 0),
        "win_rate": test_result.get('winning_trades', 0) / max(test_result.get('trade_count', 1), 1),
        "max_drawdown": test_result.get('max_drawdown', 0.0),
        "avg_hold_time": test_result.get('total_hold_time', 0.0) / max(test_result.get('trade_count', 1), 1),
        "winning_trades": test_result.get('winning_trades', 0),
        "losing_trades": test_result.get('losing_trades', 0)
    }
    
    if progress_callback:
        progress_callback(100)
    
    return {
        "outputresults1": outputresults1,
        "outputresults2": outputresults2,
        "param_stability": training_result.get("param_stability", {}),
        "besttrades": training_trades,  # Training period trades
        "all_combinations": all_combinations,
        "best_combination_idx": best_idx,
        "noalgoreturn": training_result.get("noalgoreturn", 0),
        "walk_forward_mode": True,
        "batch_mode": True,
        "segments": 1,  # Simple mode has 1 segment
        "training_score": training_score,
        "walk_forward_score": walk_forward_score,
        "combined_score": (training_score * scoring_config.get("combined_score_weighting", {}).get("training_weight", 0.4) + 
                          walk_forward_score * scoring_config.get("combined_score_weighting", {}).get("walk_forward_weight", 0.6)),
        "training_trades": training_trades,  # Store training trades separately
        "walk_forward_trades": walk_forward_trades,  # Store walk-forward trades separately
        "training_period": {"start": train_start, "end": train_end},
        "test_period": {"start": test_start, "end": test_end},
        "best_sma_a": best_a,
        "best_sma_b": best_b,
        "training_metrics": training_metrics,
        "walk_forward_metrics": walk_forward_metrics
    }

