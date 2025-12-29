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
    # If drawdown exceeds threshold, score = 0. Otherwise, linear scale from 1.0 (0% drawdown) to 0.0 (at threshold)
    max_drawdown_threshold = thresholds.get("max_drawdown_bad", 0.5)  # 50% drawdown = bad
    if max_drawdown >= max_drawdown_threshold:
        max_drawdown_score = 0.0
    else:
        max_drawdown_score = 1.0 - (max_drawdown / max_drawdown_threshold)
    scores["max_drawdown"] = max_drawdown_score * weights.get("max_drawdown", 0.10)
    
    # 5. Trade Count Score (optimal range, not too few, not too many)
    trade_count_min = thresholds.get("trade_count_min", 5)
    trade_count_max = thresholds.get("trade_count_max", 50)
    trade_count_optimal = thresholds.get("trade_count_optimal", 20)
    
    if trade_count < trade_count_min:
        # Below minimum threshold: score = 0 (no partial credit)
        trade_count_score = 0.0
    elif trade_count > trade_count_max:
        # Above maximum: gradually reduce score based on how much it exceeds max
        # Use a reasonable penalty - if it's 2x the max, score approaches 0
        excess_ratio = (trade_count - trade_count_max) / trade_count_max
        trade_count_score = max(0.0, 1.0 - excess_ratio)
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
    # If std exceeds threshold, score = 0. Otherwise, linear scale from 1.0 (0% std) to 0.0 (at threshold)
    stability_threshold = thresholds.get("stability_threshold", 0.1)  # 10% std = bad
    
    # Taxed Return Stability
    if taxed_return_std >= stability_threshold:
        taxed_return_stability_score = 0.0
    else:
        taxed_return_stability_score = 1.0 - (taxed_return_std / stability_threshold)
    scores["taxed_return_stability"] = taxed_return_stability_score * weights.get("taxed_return_stability", 0.10)
    
    # Better Off Stability
    if better_off_std >= stability_threshold:
        better_off_stability_score = 0.0
    else:
        better_off_stability_score = 1.0 - (better_off_std / stability_threshold)
    scores["better_off_stability"] = better_off_stability_score * weights.get("better_off_stability", 0.05)
    
    # Win Rate Stability
    if win_rate_std >= stability_threshold:
        win_rate_stability_score = 0.0
    else:
        win_rate_stability_score = 1.0 - (win_rate_std / stability_threshold)
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


def calculate_backtest_score_breakdown(result: Dict[str, Any], scoring_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate a backtest score breakdown showing how each metric contributes to the final score.
    
    Parameters:
        result: Dictionary containing backtest results (outputresults1, outputresults2, param_stability)
        scoring_config: Dictionary containing scoring weights and thresholds
    
    Returns:
        Dictionary with breakdown of scores, raw values, normalized scores, and contributions
    """
    if "Error" in result:
        return {
            "total_score": 0.0,
            "breakdown": [],
            "raw_values": {},
            "normalized_scores": {}
        }
    
    output1 = result.get("outputresults1", {})
    output2 = result.get("outputresults2", {})
    param_stability = result.get("param_stability", {})
    
    # Extract metrics
    taxed_return = output1.get("besttaxedreturn", 0.0)
    better_off = output1.get("betteroff", 0.0)
    win_rate = output2.get("winningtradepct", 0.0)
    max_drawdown = abs(output2.get("maxdrawdown(worst trade return pct)", 0.0))
    trade_count = output1.get("besttradecount", 0)
    avg_hold_time = output2.get("average_hold_time", 0.0)
    
    # Stability metrics
    taxed_return_std = abs(param_stability.get("taxed_return_std", 0.0))
    better_off_std = abs(param_stability.get("better_off_std", 0.0))
    win_rate_std = abs(param_stability.get("win_rate_std", 0.0))
    taxed_return_max_min_diff = abs(param_stability.get("taxed_return_max_min_diff", 0.0))
    
    # Get weights and thresholds
    weights = scoring_config.get("weights", {})
    thresholds = scoring_config.get("thresholds", {})
    
    breakdown = []
    raw_values = {}
    normalized_scores = {}
    
    # 1. Taxed Return
    taxed_return_threshold = thresholds.get("taxed_return_excellent", 0.5)
    taxed_return_norm = min(1.0, max(0.0, taxed_return / taxed_return_threshold))
    taxed_return_weight = weights.get("taxed_return", 0.25)
    taxed_return_contribution = taxed_return_norm * taxed_return_weight
    raw_values["taxed_return"] = taxed_return
    normalized_scores["taxed_return"] = taxed_return_norm
    breakdown.append({
        "metric": "Taxed Return",
        "raw_value": taxed_return,
        "raw_display": f"{taxed_return:.2%}",
        "normalized_score": taxed_return_norm,
        "weight": taxed_return_weight,
        "contribution": taxed_return_contribution,
        "explanation": f"{taxed_return:.2%} return ÷ {taxed_return_threshold:.0%} threshold = {taxed_return_norm:.2f} × {taxed_return_weight:.0%} weight = {taxed_return_contribution:.3f} pts"
    })
    
    # 2. Better Off
    better_off_threshold = thresholds.get("better_off_excellent", 0.3)
    better_off_norm = min(1.0, max(0.0, better_off / better_off_threshold))
    better_off_weight = weights.get("better_off", 0.13)
    better_off_contribution = better_off_norm * better_off_weight
    raw_values["better_off"] = better_off
    normalized_scores["better_off"] = better_off_norm
    breakdown.append({
        "metric": "Better Off",
        "raw_value": better_off,
        "raw_display": f"{better_off:.2%}",
        "normalized_score": better_off_norm,
        "weight": better_off_weight,
        "contribution": better_off_contribution,
        "explanation": f"{better_off:.2%} better ÷ {better_off_threshold:.0%} threshold = {better_off_norm:.2f} × {better_off_weight:.0%} weight = {better_off_contribution:.3f} pts"
    })
    
    # 3. Win Rate
    win_rate_threshold = thresholds.get("win_rate_excellent", 0.7)
    win_rate_norm = min(1.0, max(0.0, win_rate / win_rate_threshold))
    win_rate_weight = weights.get("win_rate", 0.26)
    win_rate_contribution = win_rate_norm * win_rate_weight
    raw_values["win_rate"] = win_rate
    normalized_scores["win_rate"] = win_rate_norm
    breakdown.append({
        "metric": "Win Rate",
        "raw_value": win_rate,
        "raw_display": f"{win_rate:.2%}",
        "normalized_score": win_rate_norm,
        "weight": win_rate_weight,
        "contribution": win_rate_contribution,
        "explanation": f"{win_rate:.2%} win rate ÷ {win_rate_threshold:.0%} threshold = {win_rate_norm:.2f} × {win_rate_weight:.0%} weight = {win_rate_contribution:.3f} pts"
    })
    
    # 4. Max Drawdown
    max_drawdown_threshold = thresholds.get("max_drawdown_bad", 0.2)
    if max_drawdown >= max_drawdown_threshold:
        max_drawdown_norm = 0.0
        max_drawdown_explanation = f"{max_drawdown:.2%} drawdown ≥ {max_drawdown_threshold:.0%} bad threshold = 0.0"
    else:
        max_drawdown_norm = 1.0 - (max_drawdown / max_drawdown_threshold)
        max_drawdown_explanation = f"1.0 - ({max_drawdown:.2%} ÷ {max_drawdown_threshold:.0%}) = {max_drawdown_norm:.2f}"
    max_drawdown_weight = weights.get("max_drawdown", 0.10)
    max_drawdown_contribution = max_drawdown_norm * max_drawdown_weight
    raw_values["max_drawdown"] = max_drawdown
    normalized_scores["max_drawdown"] = max_drawdown_norm
    breakdown.append({
        "metric": "Max Drawdown",
        "raw_value": max_drawdown,
        "raw_display": f"{max_drawdown:.2%}",
        "normalized_score": max_drawdown_norm,
        "weight": max_drawdown_weight,
        "contribution": max_drawdown_contribution,
        "explanation": f"{max_drawdown_explanation} × {max_drawdown_weight:.0%} weight = {max_drawdown_contribution:.3f} pts"
    })
    
    # 5. Trade Count
    trade_count_min = thresholds.get("trade_count_min", 10)
    trade_count_max = thresholds.get("trade_count_max", 400)
    trade_count_optimal = thresholds.get("trade_count_optimal", 60)
    
    if trade_count < trade_count_min:
        # Below minimum threshold: score = 0 (no partial credit)
        trade_count_norm = 0.0
        trade_count_explanation = f"{trade_count} trades < {trade_count_min} min threshold = 0.0"
    elif trade_count > trade_count_max:
        # Above maximum: gradually reduce score based on how much it exceeds max
        excess_ratio = (trade_count - trade_count_max) / trade_count_max
        trade_count_norm = max(0.0, 1.0 - excess_ratio)
        trade_count_explanation = f"1.0 - (({trade_count} - {trade_count_max}) ÷ {trade_count_max}) = {trade_count_norm:.2f}"
    else:
        distance_from_optimal = abs(trade_count - trade_count_optimal)
        max_distance = max(trade_count_optimal - trade_count_min, trade_count_max - trade_count_optimal)
        trade_count_norm = max(0.0, 1.0 - (distance_from_optimal / max_distance))
        trade_count_explanation = f"1.0 - (|{trade_count} - {trade_count_optimal}| ÷ {max_distance}) = {trade_count_norm:.2f}"
    
    trade_count_weight = weights.get("trade_count", 0.04)
    trade_count_contribution = trade_count_norm * trade_count_weight
    raw_values["trade_count"] = trade_count
    normalized_scores["trade_count"] = trade_count_norm
    breakdown.append({
        "metric": "Trade Count",
        "raw_value": trade_count,
        "raw_display": f"{trade_count}",
        "normalized_score": trade_count_norm,
        "weight": trade_count_weight,
        "contribution": trade_count_contribution,
        "explanation": f"{trade_count_explanation} × {trade_count_weight:.0%} weight = {trade_count_contribution:.3f} pts"
    })
    
    # 6. Hold Time
    hold_time_min = thresholds.get("hold_time_min", 3)
    hold_time_max = thresholds.get("hold_time_max", 290)
    hold_time_optimal = thresholds.get("hold_time_optimal", 40)
    
    if avg_hold_time < hold_time_min or avg_hold_time > hold_time_max:
        hold_time_norm = 0.0
        hold_time_explanation = f"{avg_hold_time:.1f} days (outside {hold_time_min}-{hold_time_max} range) = 0.0"
    else:
        distance_from_optimal = abs(avg_hold_time - hold_time_optimal)
        max_distance = max(hold_time_optimal - hold_time_min, hold_time_max - hold_time_optimal)
        hold_time_norm = max(0.0, 1.0 - (distance_from_optimal / max_distance))
        hold_time_explanation = f"1.0 - (|{avg_hold_time:.1f} - {hold_time_optimal}| ÷ {max_distance:.1f}) = {hold_time_norm:.2f}"
    
    hold_time_weight = weights.get("hold_time", 0.04)
    hold_time_contribution = hold_time_norm * hold_time_weight
    raw_values["hold_time"] = avg_hold_time
    normalized_scores["hold_time"] = hold_time_norm
    breakdown.append({
        "metric": "Hold Time",
        "raw_value": avg_hold_time,
        "raw_display": f"{avg_hold_time:.1f} days",
        "normalized_score": hold_time_norm,
        "weight": hold_time_weight,
        "contribution": hold_time_contribution,
        "explanation": f"{hold_time_explanation} × {hold_time_weight:.0%} weight = {hold_time_contribution:.3f} pts"
    })
    
    # 7. Stability Scores
    stability_threshold = thresholds.get("stability_threshold", 0.08)
    
    # Taxed Return Stability
    if taxed_return_std >= stability_threshold:
        taxed_return_stability_norm = 0.0
        taxed_return_stability_explanation = f"{taxed_return_std:.4f} std ≥ {stability_threshold:.0%} bad threshold = 0.0"
    else:
        taxed_return_stability_norm = 1.0 - (taxed_return_std / stability_threshold)
        taxed_return_stability_explanation = f"1.0 - ({taxed_return_std:.4f} ÷ {stability_threshold:.0%}) = {taxed_return_stability_norm:.2f}"
    taxed_return_stability_weight = weights.get("taxed_return_stability", 0.08)
    taxed_return_stability_contribution = taxed_return_stability_norm * taxed_return_stability_weight
    raw_values["taxed_return_stability"] = taxed_return_std
    normalized_scores["taxed_return_stability"] = taxed_return_stability_norm
    breakdown.append({
        "metric": "Taxed Return Stability",
        "raw_value": taxed_return_std,
        "raw_display": f"{taxed_return_std:.4f} std dev",
        "normalized_score": taxed_return_stability_norm,
        "weight": taxed_return_stability_weight,
        "contribution": taxed_return_stability_contribution,
        "explanation": f"{taxed_return_stability_explanation} × {taxed_return_stability_weight:.0%} weight = {taxed_return_stability_contribution:.3f} pts"
    })
    
    # Better Off Stability
    if better_off_std >= stability_threshold:
        better_off_stability_norm = 0.0
        better_off_stability_explanation = f"{better_off_std:.4f} std ≥ {stability_threshold:.0%} bad threshold = 0.0"
    else:
        better_off_stability_norm = 1.0 - (better_off_std / stability_threshold)
        better_off_stability_explanation = f"1.0 - ({better_off_std:.4f} ÷ {stability_threshold:.0%}) = {better_off_stability_norm:.2f}"
    better_off_stability_weight = weights.get("better_off_stability", 0.03)
    better_off_stability_contribution = better_off_stability_norm * better_off_stability_weight
    raw_values["better_off_stability"] = better_off_std
    normalized_scores["better_off_stability"] = better_off_stability_norm
    breakdown.append({
        "metric": "Better Off Stability",
        "raw_value": better_off_std,
        "raw_display": f"{better_off_std:.4f} std dev",
        "normalized_score": better_off_stability_norm,
        "weight": better_off_stability_weight,
        "contribution": better_off_stability_contribution,
        "explanation": f"{better_off_stability_explanation} × {better_off_stability_weight:.0%} weight = {better_off_stability_contribution:.3f} pts"
    })
    
    # Win Rate Stability
    if win_rate_std >= stability_threshold:
        win_rate_stability_norm = 0.0
        win_rate_stability_explanation = f"{win_rate_std:.4f} std ≥ {stability_threshold:.0%} bad threshold = 0.0"
    else:
        win_rate_stability_norm = 1.0 - (win_rate_std / stability_threshold)
        win_rate_stability_explanation = f"1.0 - ({win_rate_std:.4f} ÷ {stability_threshold:.0%}) = {win_rate_stability_norm:.2f}"
    win_rate_stability_weight = weights.get("win_rate_stability", 0.07)
    win_rate_stability_contribution = win_rate_stability_norm * win_rate_stability_weight
    raw_values["win_rate_stability"] = win_rate_std
    normalized_scores["win_rate_stability"] = win_rate_stability_norm
    breakdown.append({
        "metric": "Win Rate Stability",
        "raw_value": win_rate_std,
        "raw_display": f"{win_rate_std:.4f} std dev",
        "normalized_score": win_rate_stability_norm,
        "weight": win_rate_stability_weight,
        "contribution": win_rate_stability_contribution,
        "explanation": f"{win_rate_stability_explanation} × {win_rate_stability_weight:.0%} weight = {win_rate_stability_contribution:.3f} pts"
    })
    
    # Calculate total
    total_contribution = sum(item["contribution"] for item in breakdown)
    weight_sum = sum(weights.values())
    if weight_sum > 0:
        normalized_score = (total_contribution / weight_sum) * 10.0
    else:
        normalized_score = 0.0
    
    final_score = max(0.0, min(10.0, normalized_score))
    
    return {
        "total_score": final_score,
        "breakdown": breakdown,
        "raw_values": raw_values,
        "normalized_scores": normalized_scores,
        "weight_sum": weight_sum,
        "total_contribution": total_contribution
    }


def get_default_scoring_config() -> Dict[str, Any]:
    """
    Get default scoring configuration with weights and thresholds.
    
    Returns:
        Dictionary with default weights and thresholds
    """
    return {
        "weights": {
            "taxed_return": 0.25,
            "better_off": 0.13,
            "win_rate": 0.26,
            "max_drawdown": 0.10,
            "trade_count": 0.04,
            "hold_time": 0.04,
            "taxed_return_stability": 0.08,
            "better_off_stability": 0.03,
            "win_rate_stability": 0.07
        },
        "thresholds": {
            "taxed_return_excellent": 0.8,  # 80% return = excellent
            "better_off_excellent": 0.3,   # 30% better = excellent
            "win_rate_excellent": 0.7,     # 70% win rate = excellent
            "max_drawdown_bad": 0.2,       # 20% drawdown = bad
            "trade_count_min": 10,
            "trade_count_max": 400,
            "trade_count_optimal": 60,
            "hold_time_min": 3,
            "hold_time_max": 290,
            "hold_time_optimal": 40,
            "stability_threshold": 0.08     # 8% std = bad
        },
        "combined_score_weighting": {
            "training_weight": 0.4,        # 40% weight for training score
            "walk_forward_weight": 0.6      # 60% weight for walk-forward score (must sum to 1.0)
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

