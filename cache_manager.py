# cache_manager.py

import pickle
import os
import hashlib
from datetime import datetime
from pathlib import Path

# Cache directory
CACHE_DIR = Path("backtest_cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_batch_cache_dir(batch_datetime=None):
    """
    Get or create a batch cache directory with datetime subfolder.
    
    Parameters:
        batch_datetime (str or None): ISO format datetime string. If None, creates new one.
    
    Returns:
        Path: Path to batch cache directory
    """
    if batch_datetime is None:
        batch_datetime = datetime.now().isoformat()
    
    # Create safe directory name from datetime
    try:
        dt = datetime.fromisoformat(batch_datetime)
        dir_name = dt.strftime("%Y%m%d_%H%M%S")
    except:
        dir_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    batch_dir = CACHE_DIR / dir_name
    batch_dir.mkdir(exist_ok=True, parents=True)
    return batch_dir

def generate_cache_key(ticker, start_date, end_date, compounding, optimization_objective, start_amount):
    """
    Generate a unique cache key for a backtest run.
    
    Parameters:
        ticker (str): Stock ticker symbol
        start_date (str or None): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        compounding (bool): Whether compounding is enabled
        optimization_objective (str): Optimization objective used
        start_amount (float): Starting amount
    
    Returns:
        str: Cache key (hash) - used for uniqueness checking
    """
    # Create a unique string from all parameters
    key_string = f"{ticker}_{start_date}_{end_date}_{compounding}_{optimization_objective}_{start_amount}"
    # Generate hash for uniqueness checking
    cache_key = hashlib.md5(key_string.encode()).hexdigest()
    return cache_key

def generate_cache_filename(ticker, start_date, end_date, compounding, optimization_objective, start_amount, cached_at=None, walk_forward_mode=False):
    """
    Generate a human-readable cache filename.
    
    Parameters:
        ticker (str): Stock ticker symbol
        start_date (str or None): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        compounding (bool): Whether compounding is enabled
        optimization_objective (str): Optimization objective used
        start_amount (float): Starting amount
        cached_at (str or None): ISO format datetime string
        walk_forward_mode (bool): Whether this is a walk-forward analysis
    
    Returns:
        str: Human-readable filename
    """
    # Sanitize ticker for filename
    safe_ticker = ticker.replace("/", "-").replace("\\", "-")
    
    # Format dates - always use actual dates from data
    # start_date should never be None at this point (algorithm.py sets it from data)
    if start_date:
        start_str = start_date.replace("-", "")
    else:
        # Fallback: use end_date as start (shouldn't happen if algorithm.py is working correctly)
        start_str = end_date.replace("-", "") if end_date else "Unknown"
    end_str = end_date.replace("-", "")
    
    # Format datetime if provided
    if cached_at:
        try:
            dt = datetime.fromisoformat(cached_at)
            dt_str = dt.strftime("%Y%m%d_%H%M%S")
        except:
            dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build filename components
    comp_str = "Comp" if compounding else "NoComp"
    opt_str = optimization_objective.replace("_", "").title()
    wf_str = "WF" if walk_forward_mode else ""
    
    # Create filename: TICKER_StartDate_EndDate_Comp_Opt_[WF_]Datetime.pkl
    if walk_forward_mode:
        filename = f"{safe_ticker}_{start_str}_{end_str}_{comp_str}_{opt_str}_WF_{dt_str}.pkl"
    else:
        filename = f"{safe_ticker}_{start_str}_{end_str}_{comp_str}_{opt_str}_{dt_str}.pkl"
    
    # Ensure filename is not too long (Windows limit is 260 chars, but we'll be conservative)
    if len(filename) > 200:
        # Truncate but keep important parts
        if walk_forward_mode:
            filename = f"{safe_ticker}_{end_str}_{opt_str}_WF_{dt_str}.pkl"
        else:
            filename = f"{safe_ticker}_{end_str}_{opt_str}_{dt_str}.pkl"
    
    return filename

def get_cache_path(cache_key, ticker=None, start_date=None, end_date=None, compounding=None, 
                   optimization_objective=None, start_amount=None, cached_at=None, walk_forward_mode=False):
    """
    Get the full path to a cache file.
    If ticker and other info provided, uses human-readable name, otherwise uses hash.
    """
    if ticker is not None:
        filename = generate_cache_filename(ticker, start_date, end_date, compounding, 
                                         optimization_objective, start_amount, cached_at, walk_forward_mode)
        return CACHE_DIR / filename
    else:
        return CACHE_DIR / f"{cache_key}.pkl"

def save_backtest_cache(ticker, start_date, end_date, compounding, optimization_objective, 
                       start_amount, all_combinations, best_combination_idx, noalgoreturn, besttrades=None,
                       walk_forward_mode=False, segments=0, training_score=0.0, walk_forward_score=0.0, combined_score=0.0,
                       training_metrics=None, walk_forward_metrics=None, walk_forward_segment_trades=None, batch_dir=None):
    """
    Save all combination results to cache.
    
    Parameters:
        ticker (str): Stock ticker symbol
        start_date (str or None): Start date
        end_date (str): End date
        compounding (bool): Compounding flag
        optimization_objective (str): Optimization objective
        start_amount (float): Starting amount
        all_combinations (list): List of dicts, each containing combination results
        best_combination_idx (int): Index of best combination
        noalgoreturn (float): No-algorithm return value
    """
    cache_key = generate_cache_key(ticker, start_date, end_date, compounding, 
                                  optimization_objective, start_amount)
    
    cached_at = datetime.now().isoformat()
    
    # Check if a cache file with the same key already exists and delete it
    # This prevents duplicate cache files for the same parameters
    # Also check by parameters directly in case cache_key is missing
    for existing_file in CACHE_DIR.glob("*.pkl"):
        try:
            with open(existing_file, 'rb') as f:
                existing_data = pickle.load(f)
            existing_key = existing_data.get('cache_key')
            
            # Check by cache_key first (most reliable)
            if existing_key and existing_key == cache_key:
                existing_file.unlink()
                break
            
            # Fallback: check by parameters directly
            if (existing_data.get('ticker') == ticker and
                existing_data.get('start_date') == start_date and
                existing_data.get('end_date') == end_date and
                existing_data.get('compounding') == compounding and
                existing_data.get('optimization_objective') == optimization_objective and
                existing_data.get('start_amount') == start_amount):
                existing_file.unlink()
                break
        except Exception:
            # Skip corrupted files
            continue
    
    # Use human-readable filename
    if batch_dir is not None:
        # Save in batch directory
        filename = generate_cache_filename(ticker, start_date, end_date, compounding,
                                          optimization_objective, start_amount, cached_at, walk_forward_mode)
        cache_path = batch_dir / filename
    else:
        # Save in main cache directory (backward compatible)
        cache_path = get_cache_path(cache_key, ticker, start_date, end_date, compounding,
                                   optimization_objective, start_amount, cached_at, walk_forward_mode)
    
    cache_data = {
        'ticker': ticker,
        'start_date': start_date,
        'end_date': end_date,
        'compounding': compounding,
        'optimization_objective': optimization_objective,
        'start_amount': start_amount,
        'all_combinations': all_combinations,
        'best_combination_idx': best_combination_idx,
        'noalgoreturn': noalgoreturn,
        'besttrades': besttrades if besttrades else [],  # Store trades for best combination
        'cached_at': cached_at,
        'cache_key': cache_key,  # Store hash for lookup purposes
        'walk_forward_mode': walk_forward_mode,
        'segments': segments,
        'training_score': training_score,
        'walk_forward_score': walk_forward_score,
        'combined_score': combined_score,
        'training_metrics': training_metrics if training_metrics else {},
        'walk_forward_metrics': walk_forward_metrics if walk_forward_metrics else {},
        'walk_forward_segment_trades': walk_forward_segment_trades if walk_forward_segment_trades else []  # Store trades per segment
    }
    
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        return True
    except Exception as e:
        print(f"Error saving cache: {e}")
        return False

def load_backtest_cache(ticker, start_date, end_date, compounding, optimization_objective, start_amount):
    """
    Load cached backtest results if available.
    Searches for cache files matching the parameters (by hash key stored in file).
    
    Returns:
        dict or None: Cached data if found, None otherwise
    """
    cache_key = generate_cache_key(ticker, start_date, end_date, compounding, 
                                  optimization_objective, start_amount)
    
    # Search all cache files for matching cache_key
    for cache_file in CACHE_DIR.glob("*.pkl"):
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check if this cache file matches our parameters
            stored_key = cache_data.get('cache_key')
            if stored_key == cache_key:
                return cache_data
            
            # Fallback: check parameters directly if cache_key not found (old format)
            if stored_key is None:
                if (cache_data.get('ticker') == ticker and
                    cache_data.get('start_date') == start_date and
                    cache_data.get('end_date') == end_date and
                    cache_data.get('compounding') == compounding and
                    cache_data.get('optimization_objective') == optimization_objective and
                    cache_data.get('start_amount') == start_amount):
                    return cache_data
        except Exception:
            # Skip corrupted files
            continue
    
    return None

def clear_cache():
    """Clear all cached backtest results."""
    try:
        for cache_file in CACHE_DIR.glob("*.pkl"):
            cache_file.unlink()
        return True
    except Exception as e:
        print(f"Error clearing cache: {e}")
        return False

def get_cache_size():
    """Get total size of cache directory in bytes."""
    total_size = 0
    for cache_file in CACHE_DIR.glob("*.pkl"):
        total_size += cache_file.stat().st_size
    return total_size

