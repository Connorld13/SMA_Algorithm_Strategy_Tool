# scoring.py

import numpy as np
from typing import Dict, Any

def calculate_backtest_score(result: Dict[str, Any], scoring_config: Dict[str, Any]) -> float:
    """
    Calculate a backtest score from 0.0 to 10.0 based on multiple metrics.
    
    Parameters:
        result: Dictionary containing backtest results (outputresults1, outputresults2, param_stability)
        scoring_config: Dictionary containing scoring weights and thresholds
    
    Returns:
        float: Score from 0.0 to 10.0
    """
    if "Error" in result:
        return 0.0
    
    output1 = result.get("outputresults1", {})
    output2 = result.get("outputresults2", {})
    param_stability = result.get("param_stability", {})
    
    # Extract metrics
    taxed_return = output1.get("besttaxedreturn", 0.0)
    better_off = output1.get("betteroff", 0.0)
    win_rate = output2.get("winningtradepct", 0.0)
    max_drawdown = abs(output2.get("maxdrawdown(worst trade return pct)", 0.0))  # Use absolute value
    trade_count = output1.get("besttradecount", 0)
    avg_hold_time = output2.get("average_hold_time", 0.0)
    
    # Stability metrics (lower is better for std and max-min diff)
    taxed_return_std = abs(param_stability.get("taxed_return_std", 0.0))
    better_off_std = abs(param_stability.get("better_off_std", 0.0))
    win_rate_std = abs(param_stability.get("win_rate_std", 0.0))
    taxed_return_max_min_diff = abs(param_stability.get("taxed_return_max_min_diff", 0.0))
    
    # Get weights from config
    weights = scoring_config.get("weights", {})
    thresholds = scoring_config.get("thresholds", {})
    
    # Normalize each metric to 0-1 scale, then apply weights
    scores = {}
    
    # 1. Taxed Return Score (higher is better)
    taxed_return_threshold = thresholds.get("taxed_return_excellent", 0.5)  # 50% return = excellent
    taxed_return_score = min(1.0, max(0.0, taxed_return / taxed_return_threshold))
    scores["taxed_return"] = taxed_return_score * weights.get("taxed_return", 0.25)
    
    # 2. Better Off Score (higher is better)
    better_off_threshold = thresholds.get("better_off_excellent", 0.3)  # 30% better = excellent
    better_off_score = min(1.0, max(0.0, better_off / better_off_threshold))
    scores["better_off"] = better_off_score * weights.get("better_off", 0.20)
    
    # 3. Win Rate Score (higher is better)
    win_rate_threshold = thresholds.get("win_rate_excellent", 0.6)  # 60% win rate = excellent
    win_rate_score = min(1.0, max(0.0, win_rate / win_rate_threshold))
    scores["win_rate"] = win_rate_score * weights.get("win_rate", 0.15)
    
    # 4. Max Drawdown Score (lower is better, so invert)
    max_drawdown_threshold = thresholds.get("max_drawdown_bad", 0.5)  # 50% drawdown = bad
    max_drawdown_score = max(0.0, 1.0 - (max_drawdown / max_drawdown_threshold))
    scores["max_drawdown"] = max_drawdown_score * weights.get("max_drawdown", 0.10)
    
    # 5. Trade Count Score (optimal range, not too few, not too many)
    trade_count_min = thresholds.get("trade_count_min", 5)
    trade_count_max = thresholds.get("trade_count_max", 50)
    trade_count_optimal = thresholds.get("trade_count_optimal", 20)
    
    if trade_count < trade_count_min:
        trade_count_score = trade_count / trade_count_min
    elif trade_count > trade_count_max:
        trade_count_score = max(0.0, 1.0 - ((trade_count - trade_count_max) / trade_count_max))
    else:
        # Within range, score based on proximity to optimal
        distance_from_optimal = abs(trade_count - trade_count_optimal)
        max_distance = max(trade_count_optimal - trade_count_min, trade_count_max - trade_count_optimal)
        trade_count_score = max(0.0, 1.0 - (distance_from_optimal / max_distance))
    
    scores["trade_count"] = trade_count_score * weights.get("trade_count", 0.05)
    
    # 6. Average Hold Time Score (optimal range)
    hold_time_min = thresholds.get("hold_time_min", 10)
    hold_time_max = thresholds.get("hold_time_max", 365)
    hold_time_optimal = thresholds.get("hold_time_optimal", 90)
    
    if avg_hold_time < hold_time_min or avg_hold_time > hold_time_max:
        hold_time_score = 0.0
    else:
        distance_from_optimal = abs(avg_hold_time - hold_time_optimal)
        max_distance = max(hold_time_optimal - hold_time_min, hold_time_max - hold_time_optimal)
        hold_time_score = max(0.0, 1.0 - (distance_from_optimal / max_distance))
    
    scores["hold_time"] = hold_time_score * weights.get("hold_time", 0.05)
    
    # 7. Stability Scores (lower std and max-min diff is better)
    # Taxed Return Stability
    stability_threshold = thresholds.get("stability_threshold", 0.1)  # 10% std = bad
    taxed_return_stability_score = max(0.0, 1.0 - (taxed_return_std / stability_threshold))
    scores["taxed_return_stability"] = taxed_return_stability_score * weights.get("taxed_return_stability", 0.10)
    
    # Better Off Stability
    better_off_stability_score = max(0.0, 1.0 - (better_off_std / stability_threshold))
    scores["better_off_stability"] = better_off_stability_score * weights.get("better_off_stability", 0.05)
    
    # Win Rate Stability
    win_rate_stability_score = max(0.0, 1.0 - (win_rate_std / stability_threshold))
    scores["win_rate_stability"] = win_rate_stability_score * weights.get("win_rate_stability", 0.05)
    
    # Total score (sum of all weighted scores, scaled to 0-10)
    total_score = sum(scores.values())
    
    # Normalize to 0-10 range (assuming weights sum to 1.0)
    weight_sum = sum(weights.values())
    if weight_sum > 0:
        normalized_score = (total_score / weight_sum) * 10.0
    else:
        normalized_score = 0.0
    
    return max(0.0, min(10.0, normalized_score))


def get_default_scoring_config() -> Dict[str, Any]:
    """
    Get default scoring configuration with weights and thresholds.
    
    Returns:
        Dictionary with default weights and thresholds
    """
    return {
        "weights": {
            "taxed_return": 0.25,
            "better_off": 0.20,
            "win_rate": 0.15,
            "max_drawdown": 0.10,
            "trade_count": 0.05,
            "hold_time": 0.05,
            "taxed_return_stability": 0.10,
            "better_off_stability": 0.05,
            "win_rate_stability": 0.05
        },
        "thresholds": {
            "taxed_return_excellent": 0.5,  # 50% return = excellent
            "better_off_excellent": 0.3,   # 30% better = excellent
            "win_rate_excellent": 0.6,     # 60% win rate = excellent
            "max_drawdown_bad": 0.5,       # 50% drawdown = bad
            "trade_count_min": 5,
            "trade_count_max": 50,
            "trade_count_optimal": 20,
            "hold_time_min": 10,
            "hold_time_max": 365,
            "hold_time_optimal": 90,
            "stability_threshold": 0.1     # 10% std = bad
        }
    }


def rescore_cached_backtest(cache_data: Dict[str, Any], scoring_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rescore a cached backtest with new scoring configuration.
    Note: This function doesn't modify the cache file, just returns updated data.
    
    Parameters:
        cache_data: Cached backtest data
        scoring_config: New scoring configuration
    
    Returns:
        Updated cache data with new scores
    """
    all_combinations = cache_data.get("all_combinations", [])
    if not all_combinations:
        return cache_data
    
    best_idx = cache_data.get("best_combination_idx", 0)
    param_stability = cache_data.get("param_stability", {})
    
    # Calculate scores for all combinations
    all_scores = []
    for combo in all_combinations:
        combo_result = {
            "outputresults1": {
                "besttaxedreturn": combo.get("taxed_return", 0),
                "betteroff": combo.get("better_off", 0),
                "besttradecount": combo.get("trade_count", 0),
                "noalgoreturn": cache_data.get("noalgoreturn", 0)
            },
            "outputresults2": {
                "winningtradepct": combo.get("win_rate", 0),
                "maxdrawdown(worst trade return pct)": combo.get("max_drawdown", 0),
                "average_hold_time": combo.get("avg_hold_time", 0)
            },
            "param_stability": {
                "taxed_return_std": param_stability.get("taxed_return_std", 0),
                "better_off_std": param_stability.get("better_off_std", 0),
                "win_rate_std": param_stability.get("win_rate_std", 0),
                "taxed_return_max_min_diff": param_stability.get("taxed_return_max_min_diff", 0)
            }
        }
        score = calculate_backtest_score(combo_result, scoring_config)
        all_scores.append(score)
    
    # Calculate score for best combination
    best_score = all_scores[best_idx] if best_idx < len(all_scores) else 0.0
    
    # Add scores to cache data (don't modify original, create copy)
    updated_cache = cache_data.copy()
    updated_cache["score"] = best_score
    updated_cache["all_scores"] = all_scores
    updated_cache["scoring_config"] = scoring_config
    
    return updated_cache

