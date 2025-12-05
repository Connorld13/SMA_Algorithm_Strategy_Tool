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
    
    # Calculate periods in days
    backtest_period = relativedelta(years=backtest_period_years, months=backtest_period_months)
    walk_forward_period = relativedelta(years=walk_forward_period_years, months=walk_forward_period_months)
    rebalance_period = relativedelta(years=rebalance_years, months=rebalance_months) if not rebalance_none else None
    
    # Generate walk-forward segments
    segments = []
    current_end = end_date_dt
    
    while True:
        # Calculate test period end (current_end) and start
        test_end = current_end
        test_start = test_end - walk_forward_period
        
        # Calculate training period
        train_end = test_start - timedelta(days=1)  # End training day before test starts
        train_start = train_end - backtest_period
        
        # Check if we have enough data
        if train_start < data['Date'].min():
            break
        
        segments.append({
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end
        })
        
        # Move to next segment
        if rebalance_none:
            # Only do initial optimization, then walk forward without re-optimizing
            break
        elif rebalance_period is None:
            # No rebalancing - only initial segment
            break
        else:
            # Move back by rebalance period
            current_end = test_start - timedelta(days=1)
    
    if len(segments) == 0:
        raise ValueError("Not enough historical data for walk-forward analysis")
    
    # Reverse segments to process chronologically (oldest first)
    segments.reverse()
    
    print(f"Walk-forward analysis: {len(segments)} segments")
    for i, seg in enumerate(segments):
        print(f"  Segment {i+1}: Train {seg['train_start'].date()} to {seg['train_end'].date()}, "
              f"Test {seg['test_start'].date()} to {seg['test_end'].date()}")
    
    # Track aggregate results for each SMA pair
    # Key: (sma_a, sma_b), Value: aggregated metrics
    aggregate_results = {}
    best_pairs_per_segment = []
    
    # Track training and walk-forward scores separately
    training_scores = []  # Scores from optimization periods
    walk_forward_scores = []  # Scores from test periods
    
    # Store walk-forward trades per segment
    walk_forward_segment_trades = []  # List of dicts with segment info and trades
    
    # Process each segment
    total_segments = len(segments)
    for seg_idx, segment in enumerate(segments):
        if progress_callback:
            segment_progress = (seg_idx / total_segments) * 90  # 90% for segments, 10% for final aggregation
            progress_callback(segment_progress)
        
        # Get training data
        train_data = data[(data['Date'] >= segment['train_start']) & 
                         (data['Date'] <= segment['train_end'])].copy()
        
        if len(train_data) == 0:
            continue
        
        # Run optimization on training period
        train_result = algorithm.run_algorithm(
            train_data,
            start_amount=start_amount,
            progress_callback=None,  # Don't show progress for individual optimizations
            compounding=compounding,
            optimization_objective=optimization_objective,
            start_date=segment['train_start'].strftime("%Y-%m-%d"),
            end_date=segment['train_end'].strftime("%Y-%m-%d"),
            use_cache=True
        )
        
        if "Error" in train_result:
            continue
        
        # Get best SMA pair from training
        best_a = train_result['outputresults1']['besta']
        best_b = train_result['outputresults1']['bestb']
        best_pairs_per_segment.append((best_a, best_b))
        
        # Calculate training period score
        try:
            training_score = scoring.calculate_backtest_score(train_result, scoring_config)
            training_scores.append(training_score)
        except Exception as e:
            print(f"Error calculating training score: {e}")
            training_score = 0.0
            training_scores.append(0.0)
        
        # Get test data
        test_data = data[(data['Date'] >= segment['test_start']) & 
                        (data['Date'] <= segment['test_end'])].copy()
        
        if len(test_data) == 0:
            continue
        
        # Test the best pair on test period
        # We need to run the algorithm with fixed parameters
        test_result = run_fixed_parameters_backtest(
            test_data,
            sma_a=best_a,
            sma_b=best_b,
            start_amount=start_amount,
            compounding=compounding,
            start_date=segment['test_start'].strftime("%Y-%m-%d"),
            end_date=segment['test_end'].strftime("%Y-%m-%d")
        )
        
        if test_result is None:
            continue
        
        # Store trades for this segment
        segment_trades = test_result.get('trades', [])
        if segment_trades:
            # Add segment metadata to each trade for display
            for trade in segment_trades:
                trade['Segment'] = seg_idx + 1
                trade['TestStart'] = segment['test_start']
                trade['TestEnd'] = segment['test_end']
                trade['TrainStart'] = segment['train_start']
                trade['TrainEnd'] = segment['train_end']
                trade['SMA_A'] = best_a
                trade['SMA_B'] = best_b
        
        walk_forward_segment_trades.append({
            'segment': seg_idx + 1,
            'train_start': segment['train_start'],
            'train_end': segment['train_end'],
            'test_start': segment['test_start'],
            'test_end': segment['test_end'],
            'sma_a': best_a,
            'sma_b': best_b,
            'trades': segment_trades
        })
        
        # Calculate walk-forward test period score
        # Convert test_result to format expected by scoring function
        test_result_formatted = {
            "outputresults1": {
                "besttaxedreturn": test_result.get('return', 0.0),
                "betteroff": 0.0,  # Would need buy-and-hold for test period
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
        
        try:
            walk_forward_score = scoring.calculate_backtest_score(test_result_formatted, scoring_config)
            walk_forward_scores.append(walk_forward_score)
        except Exception as e:
            print(f"Error calculating walk-forward score: {e}")
            walk_forward_score = 0.0
            walk_forward_scores.append(0.0)
        
        # Aggregate results for this SMA pair
        pair_key = (best_a, best_b)
        if pair_key not in aggregate_results:
            aggregate_results[pair_key] = {
                'sma_a': best_a,
                'sma_b': best_b,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_return': 0.0,
                'cumulative_pnl': 0.0,
                'max_drawdown': 0.0,
                'total_hold_time': 0.0,
                'segments_used': 0,
                'all_returns': [],
                'all_drawdowns': []
            }
        
        agg = aggregate_results[pair_key]
        agg['segments_used'] += 1
        agg['total_trades'] += test_result.get('trade_count', 0)
        agg['winning_trades'] += test_result.get('winning_trades', 0)
        agg['losing_trades'] += test_result.get('losing_trades', 0)
        agg['total_hold_time'] += test_result.get('total_hold_time', 0.0)
        agg['cumulative_pnl'] += test_result.get('pnl', 0.0)
        agg['all_returns'].append(test_result.get('return', 0.0))
        agg['all_drawdowns'].append(abs(test_result.get('max_drawdown', 0.0)))
        
        if test_result.get('max_drawdown', 0.0) < agg['max_drawdown']:
            agg['max_drawdown'] = test_result.get('max_drawdown', 0.0)
    
    # Calculate aggregate metrics
    for pair_key, agg in aggregate_results.items():
        if agg['total_trades'] > 0:
            agg['win_rate'] = agg['winning_trades'] / agg['total_trades']
            agg['avg_hold_time'] = agg['total_hold_time'] / agg['total_trades']
        else:
            agg['win_rate'] = 0.0
            agg['avg_hold_time'] = 0.0
        
        # Calculate aggregate return
        if len(agg['all_returns']) > 0:
            # Compound returns across segments
            total_return = 1.0
            for ret in agg['all_returns']:
                total_return *= (1 + ret)
            agg['total_return'] = total_return - 1.0
        else:
            agg['total_return'] = 0.0
        
        # Average drawdown
        if len(agg['all_drawdowns']) > 0:
            agg['avg_drawdown'] = np.mean(agg['all_drawdowns'])
        else:
            agg['avg_drawdown'] = 0.0
    
    # Calculate aggregate walk-forward score (average across all test segments)
    avg_walk_forward_score = np.mean(walk_forward_scores) if walk_forward_scores else 0.0
    avg_training_score = np.mean(training_scores) if training_scores else 0.0
    
    # Calculate scores for each pair (aggregate across all segments)
    all_combinations = []
    for pair_key, agg in aggregate_results.items():
        # Create result format for scoring
        result = {
            "outputresults1": {
                "besttaxedreturn": agg['total_return'],
                "betteroff": 0.0,  # Would need buy-and-hold for each segment
                "besttradecount": agg['total_trades'],
                "noalgoreturn": 0.0
            },
            "outputresults2": {
                "winningtradepct": agg['win_rate'],
                "maxdrawdown(worst trade return pct)": agg['max_drawdown'],
                "average_hold_time": agg['avg_hold_time']
            },
            "param_stability": {
                "taxed_return_std": np.std(agg['all_returns']) if len(agg['all_returns']) > 1 else 0.0,
                "better_off_std": 0.0,
                "win_rate_std": 0.0,
                "taxed_return_max_min_diff": (max(agg['all_returns']) - min(agg['all_returns'])) if len(agg['all_returns']) > 0 else 0.0
            }
        }
        
        aggregate_score = scoring.calculate_backtest_score(result, scoring_config)
        
        all_combinations.append({
            'sma_a': agg['sma_a'],
            'sma_b': agg['sma_b'],
            'taxed_return': agg['total_return'],
            'better_off': 0.0,
            'win_rate': agg['win_rate'],
            'trade_count': agg['total_trades'],
            'winning_trades': agg['winning_trades'],
            'losing_trades': agg['losing_trades'],
            'max_drawdown': agg['max_drawdown'],
            'avg_hold_time': agg['avg_hold_time'],
            'segments_used': agg['segments_used'],
            'aggregate_score': aggregate_score,
            'training_score': avg_training_score,
            'walk_forward_score': avg_walk_forward_score,
            'combined_score': (avg_training_score * 0.4 + avg_walk_forward_score * 0.6)  # Weight: 40% training, 60% walk-forward
        })
    
    # Sort by combined score (or aggregate score if combined not available)
    all_combinations.sort(key=lambda x: x.get('combined_score', x.get('aggregate_score', 0)), reverse=True)
    
    # Find best combination
    best_combo = all_combinations[0] if all_combinations else None
    best_idx = 0
    
    # Format output similar to regular algorithm
    outputresults1 = {
        "betteroff": best_combo['better_off'] if best_combo else 0.0,
        "besttaxedreturn": best_combo['taxed_return'] if best_combo else 0.0,
        "noalgoreturn": 0.0,
        "besta": best_combo['sma_a'] if best_combo else 0,
        "bestb": best_combo['sma_b'] if best_combo else 0,
        "besttradecount": best_combo['trade_count'] if best_combo else 0,
        "avgtradepct": 0.0,
        "iterations": 0,
        "combinations": len(all_combinations),
        "optimization_objective": optimization_objective,
        "stability_range": 0,
        "stability_combinations": 0
    }
    
    outputresults2 = {
        "startamount": start_amount,
        "bestendtaxed_liquidity": start_amount * (1 + best_combo['taxed_return']) if best_combo else start_amount,
        "(noalgoreturn+1)*startamount": start_amount,
        "losingtrades": best_combo['losing_trades'] if best_combo else 0,
        "losingtradepct": 1 - best_combo['win_rate'] if best_combo else 0.0,
        "winningtradepct": best_combo['win_rate'] if best_combo else 0.0,
        "maxdrawdown(worst trade return pct)": best_combo['max_drawdown'] if best_combo else 0.0,
        "average_hold_time": best_combo['avg_hold_time'] if best_combo else 0.0,
        "win_percentage_last_4_trades": None
    }
    
    param_stability = {
        "taxed_return_avg": np.mean([c['taxed_return'] for c in all_combinations]) if all_combinations else 0.0,
        "taxed_return_std": np.std([c['taxed_return'] for c in all_combinations]) if len(all_combinations) > 1 else 0.0,
        "taxed_return_max": max([c['taxed_return'] for c in all_combinations]) if all_combinations else 0.0,
        "taxed_return_min": min([c['taxed_return'] for c in all_combinations]) if all_combinations else 0.0,
        "taxed_return_max_min_diff": 0.0,
        "taxed_return_max_avg_diff": 0.0,
        "better_off_avg": 0.0,
        "better_off_std": 0.0,
        "better_off_max": 0.0,
        "better_off_min": 0.0,
        "better_off_max_min_diff": 0.0,
        "better_off_max_avg_diff": 0.0,
        "win_rate_avg": np.mean([c['win_rate'] for c in all_combinations]) if all_combinations else 0.0,
        "win_rate_std": np.std([c['win_rate'] for c in all_combinations]) if len(all_combinations) > 1 else 0.0,
        "win_rate_max": max([c['win_rate'] for c in all_combinations]) if all_combinations else 0.0,
        "win_rate_min": min([c['win_rate'] for c in all_combinations]) if all_combinations else 0.0,
        "win_rate_max_min_diff": 0.0,
        "win_rate_max_avg_diff": 0.0,
        "trade_count_avg": np.mean([c['trade_count'] for c in all_combinations]) if all_combinations else 0.0,
        "trade_count_std": np.std([c['trade_count'] for c in all_combinations]) if len(all_combinations) > 1 else 0.0,
        "trade_count_max": max([c['trade_count'] for c in all_combinations]) if all_combinations else 0,
        "trade_count_min": min([c['trade_count'] for c in all_combinations]) if all_combinations else 0,
        "trade_count_max_min_diff": 0,
        "trade_count_max_avg_diff": 0
    }
    
    if progress_callback:
        progress_callback(100)
    
    # Calculate training and walk-forward metrics for caching
    training_metrics = {}
    walk_forward_metrics = {}
    
    if best_combo:
        # Aggregate training metrics (from all training periods)
        training_metrics = {
            "taxed_return": best_combo.get('taxed_return', 0.0),  # This is actually walk-forward return
            "better_off": 0.0,
            "trade_count": best_combo.get('trade_count', 0),
            "win_rate": best_combo.get('win_rate', 0.0),
            "max_drawdown": best_combo.get('max_drawdown', 0.0),
            "avg_hold_time": best_combo.get('avg_hold_time', 0.0)
        }
        
        # Walk-forward metrics (same as training for now, but separated for clarity)
        walk_forward_metrics = {
            "taxed_return": best_combo.get('taxed_return', 0.0),
            "better_off": 0.0,
            "trade_count": best_combo.get('trade_count', 0),
            "win_rate": best_combo.get('win_rate', 0.0),
            "max_drawdown": best_combo.get('max_drawdown', 0.0),
            "avg_hold_time": best_combo.get('avg_hold_time', 0.0),
            "winning_trades": best_combo.get('winning_trades', 0),
            "losing_trades": best_combo.get('losing_trades', 0)
        }
    
    return {
        "outputresults1": outputresults1,
        "outputresults2": outputresults2,
        "param_stability": param_stability,
        "besttrades": [],  # Empty for walk-forward (trades stored per segment)
        "all_combinations": all_combinations,
        "best_combination_idx": best_idx,
        "noalgoreturn": 0.0,
        "walk_forward_mode": True,
        "segments": len(segments),
        "best_pairs_per_segment": best_pairs_per_segment,
        "training_score": avg_training_score,
        "walk_forward_score": avg_walk_forward_score,
        "combined_score": (avg_training_score * 0.4 + avg_walk_forward_score * 0.6) if (training_scores and walk_forward_scores) else 0.0,
        "walk_forward_segment_trades": walk_forward_segment_trades,  # Store trades per segment
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
    Run walk-forward analysis using the best scored combo from a training period.
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
    
    # Calculate periods
    backtest_period = relativedelta(years=backtest_period_years, months=backtest_period_months)
    walk_forward_period = relativedelta(years=walk_forward_period_years, months=walk_forward_period_months)
    
    # Generate walk-forward segments
    segments = []
    current_end = end_date_dt
    
    while True:
        test_end = current_end
        test_start = test_end - walk_forward_period
        train_end = test_start - timedelta(days=1)
        train_start = train_end - backtest_period
        
        if train_start < data['Date'].min():
            break
        
        segments.append({
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end
        })
        
        # Move to next segment (no rebalancing for batch mode)
        current_end = test_start - timedelta(days=1)
    
    if len(segments) == 0:
        return {"Error": "Not enough historical data for walk-forward analysis"}
    
    segments.reverse()  # Process chronologically
    
    # If training_result provided, use it; otherwise run training on first segment
    if training_result is None or "Error" in training_result:
        # Run training on first segment to get best combo
        first_segment = segments[0]
        train_data = data[(data['Date'] >= first_segment['train_start']) & 
                         (data['Date'] <= first_segment['train_end'])].copy()
        
        if len(train_data) == 0:
            return {"Error": "No training data available"}
        
        training_result = algorithm.run_algorithm(
            train_data,
            start_amount=start_amount,
            progress_callback=None,
            compounding=compounding,
            optimization_objective=optimization_objective,
            start_date=first_segment['train_start'].strftime("%Y-%m-%d"),
            end_date=first_segment['train_end'].strftime("%Y-%m-%d"),
            use_cache=True
        )
    
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
    
    # Calculate training score
    training_score = best_score
    
    # Now run walk-forward tests using this best combo
    walk_forward_results = []
    walk_forward_scores = []
    walk_forward_segment_trades = []  # Store trades per segment
    
    total_segments = len(segments)
    for seg_idx, segment in enumerate(segments):
        if progress_callback:
            segment_progress = (seg_idx / total_segments) * 90
            progress_callback(segment_progress)
        
        # Get test data
        test_data = data[(data['Date'] >= segment['test_start']) & 
                        (data['Date'] <= segment['test_end'])].copy()
        
        if len(test_data) == 0:
            continue
        
        # Test the best combo on test period
        test_result = run_fixed_parameters_backtest(
            test_data,
            sma_a=best_a,
            sma_b=best_b,
            start_amount=start_amount,
            compounding=compounding,
            start_date=segment['test_start'].strftime("%Y-%m-%d"),
            end_date=segment['test_end'].strftime("%Y-%m-%d")
        )
        
        if test_result is None:
            continue
        
        # Store trades for this segment
        segment_trades = test_result.get('trades', [])
        if segment_trades:
            # Add segment metadata to each trade for display
            for trade in segment_trades:
                trade['Segment'] = seg_idx + 1
                trade['TestStart'] = segment['test_start']
                trade['TestEnd'] = segment['test_end']
                trade['TrainStart'] = segment['train_start']
                trade['TrainEnd'] = segment['train_end']
                trade['SMA_A'] = best_a
                trade['SMA_B'] = best_b
        
        walk_forward_segment_trades.append({
            'segment': seg_idx + 1,
            'train_start': segment['train_start'],
            'train_end': segment['train_end'],
            'test_start': segment['test_start'],
            'test_end': segment['test_end'],
            'sma_a': best_a,
            'sma_b': best_b,
            'trades': segment_trades
        })
        
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
        
        wf_score = scoring.calculate_backtest_score(test_result_formatted, scoring_config)
        walk_forward_scores.append(wf_score)
        walk_forward_results.append(test_result)
    
    # Calculate aggregate walk-forward metrics
    if not walk_forward_results:
        return {"Error": "No walk-forward test results"}
    
    avg_walk_forward_score = sum(walk_forward_scores) / len(walk_forward_scores) if walk_forward_scores else 0.0
    
    # Aggregate walk-forward metrics
    total_wf_trades = sum(r.get('trade_count', 0) for r in walk_forward_results)
    total_wf_wins = sum(r.get('winning_trades', 0) for r in walk_forward_results)
    total_wf_losses = sum(r.get('losing_trades', 0) for r in walk_forward_results)
    total_wf_return = sum(r.get('return', 0.0) for r in walk_forward_results) / len(walk_forward_results)
    total_wf_pnl = sum(r.get('pnl', 0.0) for r in walk_forward_results)
    wf_max_drawdown = min([r.get('max_drawdown', 0.0) for r in walk_forward_results])
    avg_wf_hold_time = sum(r.get('total_hold_time', 0.0) for r in walk_forward_results) / max(total_wf_trades, 1)
    
    # Format results similar to regular walk forward
    outputresults1 = {
        "besta": best_a,
        "bestb": best_b,
        "besttaxedreturn": total_wf_return,
        "betteroff": 0.0,
        "besttradecount": total_wf_trades,
        "noalgoreturn": 0.0,
        "optimization_objective": optimization_objective
    }
    
    outputresults2 = {
        "winningtradepct": total_wf_wins / max(total_wf_trades, 1),
        "maxdrawdown(worst trade return pct)": wf_max_drawdown,
        "average_hold_time": avg_wf_hold_time
    }
    
    if progress_callback:
        progress_callback(100)
    
    return {
        "outputresults1": outputresults1,
        "outputresults2": outputresults2,
        "param_stability": training_result.get("param_stability", {}),
        "besttrades": [],
        "all_combinations": all_combinations,
        "best_combination_idx": best_idx,
        "noalgoreturn": training_result.get("noalgoreturn", 0),
        "walk_forward_mode": True,
        "batch_mode": True,
        "segments": len(segments),
        "training_score": training_score,
        "walk_forward_score": avg_walk_forward_score,
        "training_metrics": {
            "taxed_return": best_combo.get("taxed_return", 0),
            "better_off": best_combo.get("better_off", 0),
            "win_rate": best_combo.get("win_rate", 0),
            "trade_count": best_combo.get("trade_count", 0),
            "max_drawdown": best_combo.get("max_drawdown", 0),
            "avg_hold_time": best_combo.get("avg_hold_time", 0)
        },
        "walk_forward_metrics": {
            "taxed_return": total_wf_return,
            "win_rate": total_wf_wins / max(total_wf_trades, 1),
            "trade_count": total_wf_trades,
            "winning_trades": total_wf_wins,
            "losing_trades": total_wf_losses,
            "max_drawdown": wf_max_drawdown,
            "avg_hold_time": avg_wf_hold_time,
            "total_pnl": total_wf_pnl
        },
        "walk_forward_segment_trades": walk_forward_segment_trades  # Store trades per segment
    }

