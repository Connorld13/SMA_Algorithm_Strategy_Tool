# view_cache.py
# Utility script to view cached backtest results

import pickle
import pandas as pd
from pathlib import Path
import json

CACHE_DIR = Path("backtest_cache")

def view_cache_file(cache_file_path):
    """View the contents of a cache file."""
    try:
        with open(cache_file_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        print("=" * 80)
        print(f"Cache File: {cache_file_path.name}")
        print("=" * 80)
        print(f"\nTicker: {cache_data.get('ticker', 'N/A')}")
        print(f"Start Date: {cache_data.get('start_date', 'N/A')}")
        print(f"End Date: {cache_data.get('end_date', 'N/A')}")
        print(f"Compounding: {cache_data.get('compounding', 'N/A')}")
        print(f"Optimization Objective: {cache_data.get('optimization_objective', 'N/A')}")
        print(f"Start Amount: ${cache_data.get('start_amount', 0):,.2f}")
        print(f"Cached At: {cache_data.get('cached_at', 'N/A')}")
        print(f"No Algorithm Return: {cache_data.get('noalgoreturn', 0):.4%}")
        print(f"Best Combination Index: {cache_data.get('best_combination_idx', 'N/A')}")
        
        all_combinations = cache_data.get('all_combinations', [])
        print(f"\nTotal Combinations: {len(all_combinations)}")
        
        if all_combinations:
            # Show best combination
            best_idx = cache_data.get('best_combination_idx', 0)
            if 0 <= best_idx < len(all_combinations):
                best = all_combinations[best_idx]
                print("\n" + "=" * 80)
                print("BEST COMBINATION:")
                print("=" * 80)
                print(f"SMA A: {best.get('sma_a', 'N/A')}")
                print(f"SMA B: {best.get('sma_b', 'N/A')}")
                print(f"Taxed Return: {best.get('taxed_return', 0):.4%}")
                print(f"Better Off: {best.get('better_off', 0):.4%}")
                print(f"Win Rate: {best.get('win_rate', 0):.2%}")
                print(f"Trade Count: {best.get('trade_count', 0)}")
                print(f"Winning Trades: {best.get('winning_trades', 0)}")
                print(f"Losing Trades: {best.get('losing_trades', 0)}")
                print(f"Max Drawdown: {best.get('max_drawdown', 0):.4%}")
                print(f"Avg Hold Time: {best.get('avg_hold_time', 0):.1f} days")
                print(f"Avg Trade Return: {best.get('avg_trade_return', 0):.4%}")
                print(f"Return Std Dev: {best.get('return_std', 0):.4%}")
                print(f"Win % Last 4 Trades: {best.get('win_pct_last_4', 'N/A')}")
            
            # Show top 10 combinations by taxed return
            print("\n" + "=" * 80)
            print("TOP 10 COMBINATIONS BY TAXED RETURN:")
            print("=" * 80)
            sorted_combos = sorted(all_combinations, key=lambda x: x.get('taxed_return', -999), reverse=True)
            print(f"{'Rank':<6} {'SMA A':<8} {'SMA B':<8} {'Taxed Return':<15} {'Win Rate':<12} {'Trades':<8} {'Better Off':<12}")
            print("-" * 80)
            for i, combo in enumerate(sorted_combos[:10], 1):
                print(f"{i:<6} {combo.get('sma_a', 0):<8} {combo.get('sma_b', 0):<8} "
                      f"{combo.get('taxed_return', 0):>13.4%} {combo.get('win_rate', 0):>11.2%} "
                      f"{combo.get('trade_count', 0):<8} {combo.get('better_off', 0):>11.4%}")
            
            # Option to export to CSV
            print("\n" + "=" * 80)
            export = input("Export all combinations to CSV? (y/n): ").strip().lower()
            if export == 'y':
                df = pd.DataFrame(all_combinations)
                output_file = f"cache_export_{cache_file_path.stem}.csv"
                df.to_csv(output_file, index=False)
                print(f"Exported to {output_file}")
        
        return cache_data
        
    except Exception as e:
        print(f"Error loading cache file: {e}")
        return None

def list_cache_files():
    """List all cache files."""
    cache_files = list(CACHE_DIR.glob("*.pkl"))
    if not cache_files:
        print("No cache files found.")
        return []
    
    print(f"\nFound {len(cache_files)} cache file(s):\n")
    for i, cache_file in enumerate(cache_files, 1):
        file_size = cache_file.stat().st_size / 1024  # Size in KB
        print(f"{i}. {cache_file.name} ({file_size:.2f} KB)")
    
    return cache_files

if __name__ == "__main__":
    print("Backtest Cache Viewer")
    print("=" * 80)
    
    cache_files = list_cache_files()
    
    if cache_files:
        if len(cache_files) == 1:
            # If only one file, view it directly
            view_cache_file(cache_files[0])
        else:
            # Let user choose
            choice = input(f"\nSelect cache file to view (1-{len(cache_files)}) or 'all' for all: ").strip()
            
            if choice.lower() == 'all':
                for cache_file in cache_files:
                    view_cache_file(cache_file)
                    print("\n")
            else:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(cache_files):
                        view_cache_file(cache_files[idx])
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Invalid input.")

