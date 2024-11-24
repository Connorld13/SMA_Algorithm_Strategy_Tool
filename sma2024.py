import numpy as np
import pandas as pd
from datetime import datetime

# Load and clean the data
data = pd.read_csv('data.csv')

# Reverse the data order to match MATLAB's flip function and reset the index
data = data.iloc[::-1].reset_index(drop=True)

# Convert 'Date' to datetime and set it as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Select and clean the stock price column
stockcol = 'Close/Last'
# Remove '$', split at '.', take the integer part, and convert to float
data[stockcol] = data[stockcol].str.replace('$', '', regex=False).str.split('.').str[0].astype(float)
prices = data[stockcol].values
dates = data.index

# Parameters
startamount = 10000
over1yeartax = 0.78
under1yeartax = 0.65

# Define ranges for SMAs
astart, aend, bstart, bend, inc = 5, 200, 5, 200, 5
combinations = (((aend - astart) // inc) + 1) * (((bend - bstart) // inc) + 1)
iterations = 0

# Initialize best return variables
besttaxedreturn = -1
besta = None
bestb = None
besttradecount = 0
besttrades = []
bestendtaxed_liquidity = startamount

# Precompute all SMA1 and SMA2 to optimize performance
sma1_dict = {}
sma2_dict = {}

for a in range(astart, aend + 1, inc):
    sma1 = pd.Series(prices).rolling(window=a, min_periods=a).mean().values
    sma1_dict[a] = sma1

for b in range(bstart, bend + 1, inc):
    sma2 = pd.Series(prices).rolling(window=b, min_periods=b).mean().values
    sma2_dict[b] = sma2

# Initialize a list to store progress stats
progress_stats = []

# Main optimization loop
for a in range(astart, aend + 1, inc):
    for b in range(bstart, bend + 1, inc):
        sma1 = sma1_dict[a]
        sma2 = sma2_dict[b]
        numrows = len(prices)

        # Initialize buy/sell signals
        buysells = np.zeros(numrows)
        pos = 0  # 0: not in position, 1: in position

        # Adjust the starting index to match MATLAB's behavior
        start_index = b - 1  # Start from b instead of max(a, b)

        # Generate buy/sell signals starting from the b-th index
        for i in range(start_index + 1, numrows):
            # Only consider if both SMAs are valid
            if np.isnan(sma1[i]) or np.isnan(sma2[i]) or np.isnan(sma1[i-1]) or np.isnan(sma2[i-1]):
                continue  # Skip if any SMA value is NaN

            smadiff = sma1[i] - sma2[i]
            prev_diff = sma1[i-1] - sma2[i-1]
            diff_change = smadiff - prev_diff

            if diff_change > 0 and pos == 0:
                buysells[i] = 1  # Buy
                pos = 1
            elif diff_change < 0 and pos == 1:
                buysells[i] = -1  # Sell
                pos = 0
            # Else, no action

        # Record trades based on buy/sell signals
        trades = []
        buy_index = None

        for i in range(start_index, numrows):
            signal = buysells[i]
            if signal == 1:
                buy_index = i
                trades.append({
                    'TradeNumber': len(trades) + 1,
                    'Type': 'Buy',
                    'DateNum': dates[i],
                    'Price': prices[i]
                })
            elif signal == -1 and buy_index is not None:
                sell_index = i
                sell_price = prices[sell_index]
                buy_price = prices[buy_index]
                pre_tax_return = (sell_price - buy_price) / buy_price
                hold_time = (dates[sell_index] - dates[buy_index]).days
                trades.append({
                    'TradeNumber': len(trades) + 1,
                    'Type': 'Sell',
                    'DateNum': dates[sell_index],
                    'Price': sell_price,
                    'PreTaxReturn': pre_tax_return,
                    'HoldTime': hold_time
                })
                buy_index = None  # Reset after selling

        # Compute cumulative taxed liquidity based on MATLAB's additive logic
        under1yearpl = 0
        over1yearpl = 0
        pre_tax_liquidity = startamount

        for trade in trades:
            if trade['Type'] == 'Sell':
                pre_tax_return = trade['PreTaxReturn']
                hold_time = trade['HoldTime']
                profit_dollars = pre_tax_return * pre_tax_liquidity
                if hold_time < 365:
                    if profit_dollars > 0:
                        under1yearpl += profit_dollars
                    else:
                        under1yearpl += profit_dollars  # No tax on losses
                else:
                    if profit_dollars > 0:
                        over1yearpl += profit_dollars
                    else:
                        over1yearpl += profit_dollars  # No tax on losses
                pre_tax_liquidity += profit_dollars  # Update liquidity after trade

        # Calculate taxed return
        if len(trades) > 1 and (under1yearpl + over1yearpl) > 0:
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
        endtaxed_liquidity = startamount + taxcumpl
        taxcumreturn = (endtaxed_liquidity / startamount) - 1

        # Update best return stats if current is better
        if taxcumreturn > besttaxedreturn:
            besttaxedreturn = taxcumreturn
            besta = a
            bestb = b
            besttradecount = len(trades)
            besttrades = trades.copy()
            bestendtaxed_liquidity = endtaxed_liquidity

        iterations += 1

        # Capture the current 'a' and 'b' values
        current_a = a
        current_b = b

        # At every 40 iterations, output the progress stats
        if iterations % 40 == 0:
            outputstats1 = [
                besttaxedreturn,
                besta,
                bestb,
                besttradecount,
                iterations,
                combinations,
                current_a,
                current_b,
                taxcumreturn
            ]
            progress_stats.append(outputstats1)  # Optional: Store progress stats for later use

            print(f"Progress: {iterations}/{combinations} - Best Return: {besttaxedreturn:.6f}")
            print(f"outputstats1 = {outputstats1}")

# After all iterations, calculate overall stats
total_days = (dates[-1] - dates[0]).days
price_return = (prices[-1] - prices[0]) / prices[0]

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

if noalgoreturn != 0:
    betteroff = (besttaxedreturn / noalgoreturn) - 1
else:
    betteroff = 0

# Calculate losing trades and max drawdown
losingtrades = sum(1 for trade in besttrades if trade.get('PreTaxReturn', 0) < 0)
losingtradepct = (losingtrades * 2) / besttradecount if besttradecount else 0  # Multiply by 2 to match MATLAB's logic

# Average trade percentage: considering only sell trades
sell_trades = [trade for trade in besttrades if trade['Type'] == 'Sell']
avgtradepct = besttaxedreturn / len(sell_trades) if sell_trades else 0

# Max drawdown: worst trade return percentage
maxdrawdown = min(trade['PreTaxReturn'] for trade in besttrades if trade['Type'] == 'Sell') if sell_trades else 0

# Output results
outputresults1 = [
    betteroff,
    besttaxedreturn,
    noalgoreturn,
    besta,
    bestb,
    besttradecount,
    avgtradepct,
    iterations,
    combinations,
]

outputresults2 = [
    startamount,
    bestendtaxed_liquidity,
    (noalgoreturn + 1) * startamount,
    losingtrades,
    losingtradepct,
    maxdrawdown,
]

# Print results with appropriate formatting
print("\nOutput Results 1:")
print(f"betteroff={betteroff:.6f}")
print(f"besttaxedreturn={besttaxedreturn:.4f}")
print(f"noalgoreturn={noalgoreturn:.4f}")
print(f"besta={besta}")
print(f"bestb={bestb}")
print(f"besttradecount={besttradecount}")
print(f"avgtradepct={avgtradepct:.6f}")
print(f"iterations={iterations}")
print(f"combinations={combinations}")

print("\nOutput Results 2:")
print(f"startamount={startamount}")
print(f"bestendtaxed_liquidity={bestendtaxed_liquidity:.0f}")
print(f"(noalgoreturn+1)*startamount={ (noalgoreturn +1)*startamount:.0f}")
print(f"losingtrades={losingtrades}")
print(f"losingtradepct={losingtradepct:.5f}")
print(f"maxdrawdown(worst trade return pct)={maxdrawdown:.6f}")
