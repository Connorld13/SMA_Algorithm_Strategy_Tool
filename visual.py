# visual.py

import pandas as pd
import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, ColorBar
from bokeh.layouts import column, row
from bokeh.transform import transform

def show_interactive_chart(
    ticker,
    data,
    algo_result,
    param_results=None,
    show_drawdown=True,
    show_distribution=True,
    show_winloss=True,
    show_param_heatmap=True,
):
    """
    Displays an advanced interactive Bokeh dashboard with multiple charts:
      1) Main Price Chart (Candlestick or line) + Buy/Sell markers (no tooltips on markers).
      2) Equity Curve from realized trades.
      3) (Optional) Drawdown Chart.
      4) (Optional) Distribution of Trade Returns (Histogram).
      5) (Optional) Win/Loss Timeline (Trade returns in chronological order).
      6) (Optional) Parameter Heatmap (if param_results provided).

    Parameters:
        ticker (str): Stock symbol.
        data (pd.DataFrame): Historical data with 'Date' and 'Close'. If also has 'Open','High','Low', candlesticks are drawn.
        algo_result (dict): Output from run_algorithm(...), containing 'besttrades' etc.
        param_results (pd.DataFrame or None): Optional DataFrame of parameter-scan results (e.g. columns ['a','b','final_return']).
        show_drawdown (bool): Whether to include a drawdown chart.
        show_distribution (bool): Whether to include a histogram of trade returns.
        show_winloss (bool): Whether to include a timeline of trade returns in chronological order.
        show_param_heatmap (bool): Whether to include a parameter heatmap (only if param_results is not None).
    """

    # --------------------
    # 1) Basic Checks
    # --------------------
    if data is None or data.empty:
        print(f"No data to visualize for {ticker}.")
        return

    best_trades = algo_result.get("besttrades", [])
    if not pd.api.types.is_datetime64_any_dtype(data["Date"]):
        data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    # Identify if we have OHLC data
    has_ohlc = all(col in data.columns for col in ["Open", "High", "Low", "Close"])

    # We'll collect our final figures in a list and then use column(...) or row(...)
    figures_list = []

    # --------------------
    # 2) Main Price Chart
    # --------------------
    p_price = figure(
        x_axis_type="datetime",
        title=f"{ticker} Price & Signals",
        width=800,
        height=400,
        sizing_mode="stretch_width",
        toolbar_location="above",
        tools="pan,wheel_zoom,box_zoom,reset,save"  # remove default hover
    )

    price_source = ColumnDataSource(data)

    if has_ohlc:
        # Candlestick approach
        inc = data["Close"] >= data["Open"]
        dec = data["Close"] < data["Open"]

        p_price.segment(
            x0="Date",
            y0="High",
            x1="Date",
            y1="Low",
            color="black",
            source=price_source
        )
        inc_source = ColumnDataSource(data[inc])
        dec_source = ColumnDataSource(data[dec])

        p_price.vbar(
            x="Date", width=12*60*60*1000,
            top="Close", bottom="Open",
            fill_color="#a8df65", line_color="black",
            source=inc_source
        )
        p_price.vbar(
            x="Date", width=12*60*60*1000,
            top="Open", bottom="Close",
            fill_color="#f2583e", line_color="black",
            source=dec_source
        )

        # Hover for candlesticks
        candlestick_hover = HoverTool(
            tooltips=[
                ("Date", "@Date{%F}"),
                ("Open", "@Open{0.2f}"),
                ("High", "@High{0.2f}"),
                ("Low", "@Low{0.2f}"),
                ("Close", "@Close{0.2f}")
            ],
            formatters={"@Date": "datetime"},
            mode="vline"
        )
        p_price.add_tools(candlestick_hover)
    else:
        # Simple line chart if no OHLC
        line_glyph = p_price.line(
            x="Date",
            y="Close",
            source=price_source,
            line_width=2,
            color="blue"
        )
        # Hover for line
        line_hover = HoverTool(
            tooltips=[
                ("Date", "@Date{%F}"),
                ("Close", "@Close{0.2f}")
            ],
            formatters={"@Date": "datetime"},
            mode="vline",
            renderers=[line_glyph]
        )
        p_price.add_tools(line_hover)

    # Buy/Sell markers (no hover)
    if best_trades:
        buy_x, buy_y = [], []
        sell_x, sell_y = [], []
        for tr in best_trades:
            if tr["Buy/Sell"] == 1:
                buy_x.append(tr["Date"])
                buy_y.append(tr["Price"])
            elif tr["Buy/Sell"] == -1:
                sell_x.append(tr["Date"])
                sell_y.append(tr["Price"])

        p_price.scatter(buy_x, buy_y, marker="triangle", color="green", size=10, legend_label="Buy")
        p_price.scatter(sell_x, sell_y, marker="inverted_triangle", color="red", size=10, legend_label="Sell")
    else:
        print("No best trades to display for buy/sell markers.")

    p_price.legend.location = "top_left"
    figures_list.append(p_price)

    # --------------------
    # 3) Equity Curve
    # --------------------
    p_equity = figure(
        x_axis_type="datetime",
        title=f"{ticker} Equity Curve (Approx)",
        width=800,
        height=200,
        sizing_mode="stretch_width",
        toolbar_location="above",
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )

    if best_trades:
        trades_sorted = sorted(best_trades, key=lambda x: x["Date"])
        eq_dates = []
        eq_vals = []
        for t in trades_sorted:
            if t["Buy/Sell"] == -1:
                eq_dates.append(t["Date"])
                eq_vals.append(t["PreTaxLiquidity"])
        if eq_dates:
            eq_df = pd.DataFrame({"Date": eq_dates, "Equity": eq_vals})
            eq_source = ColumnDataSource(eq_df)
            eq_line = p_equity.line("Date", "Equity", source=eq_source, line_width=2, color="navy")
            
            eq_hover = HoverTool(
                tooltips=[("Date", "@Date{%F}"), ("Equity", "@Equity{0.2f}")],
                formatters={"@Date": "datetime"},
                mode="vline",
                renderers=[eq_line]
            )
            p_equity.add_tools(eq_hover)
        else:
            print("No sell trades found to build equity curve.")
    else:
        print("No trades found to build equity curve.")

    figures_list.append(p_equity)

    # --------------------------------
    # 4) Drawdown Chart (Optional)
    # --------------------------------
    if show_drawdown and best_trades:
        # We can reuse eq_df if it exists; if eq_dates was empty, we skip
        if "eq_df" in locals() and not eq_df.empty:
            # Build a drawdown column
            eq_df = eq_df.sort_values("Date")
            eq_df["RunningMax"] = eq_df["Equity"].cummax()
            eq_df["Drawdown"] = (eq_df["Equity"] - eq_df["RunningMax"]) / eq_df["RunningMax"]

            p_drawdown = figure(
                x_axis_type="datetime",
                title="Drawdown",
                width=800,
                height=200,
                sizing_mode="stretch_width",
                toolbar_location="above",
                tools="pan,wheel_zoom,box_zoom,reset,save"
            )

            dd_source = ColumnDataSource(eq_df)
            dd_line = p_drawdown.line("Date", "Drawdown", source=dd_source, line_width=2, color="firebrick")

            dd_hover = HoverTool(
                tooltips=[("Date", "@Date{%F}"), ("Drawdown", "@Drawdown{0.2%}")],
                formatters={"@Date": "datetime"},
                mode="vline",
                renderers=[dd_line]
            )
            p_drawdown.add_tools(dd_hover)

            figures_list.append(p_drawdown)
        else:
            print("Cannot plot drawdown - no equity curve data found.")

    # ------------------------------------------------
    # 5) Distribution of Trade Returns (Optional)
    # ------------------------------------------------
    if show_distribution and best_trades:
        # Get only closed trades (where Buy/Sell == -1)
        sell_trades = [t for t in best_trades if t["Buy/Sell"] == -1]
        returns = [t["PreTaxReturn"] for t in sell_trades]
        if returns:
            hist, edges = np.histogram(returns, bins=30)
            hist_df = pd.DataFrame({
                "count": hist,
                "left": edges[:-1],
                "right": edges[1:]
            })

            p_hist = figure(
                title="Distribution of Trade Returns",
                width=400,
                height=300,
                sizing_mode="stretch_width",
                toolbar_location="above",
                tools="pan,wheel_zoom,box_zoom,reset,save"
            )
            hist_src = ColumnDataSource(hist_df)
            p_hist.quad(
                top="count",
                bottom=0,
                left="left",
                right="right",
                source=hist_src,
                fill_color="blue",
                line_color="white",
                alpha=0.7
            )
            p_hist.xaxis.axis_label = "Trade Return"
            p_hist.yaxis.axis_label = "Frequency"

            figures_list.append(p_hist)
        else:
            print("No closed trades to plot distribution.")
    elif show_distribution:
        print("No best trades found to plot distribution of trade returns.")

    # ------------------------------------------------
    # 6) Win/Loss Timeline (Optional)
    # ------------------------------------------------
    if show_winloss and best_trades:
        # Get only sell trades
        sell_trades = [t for t in best_trades if t["Buy/Sell"] == -1]
        if sell_trades:
            # Chronological order
            sell_trades_sorted = sorted(sell_trades, key=lambda x: x["Date"])
            timeline_df = pd.DataFrame({
                "Index": range(1, len(sell_trades_sorted) + 1),
                "Return": [s["PreTaxReturn"] for s in sell_trades_sorted]
            })
            timeline_src = ColumnDataSource(timeline_df)

            p_winloss = figure(
                title="Win/Loss Timeline (Closed Trades in Chronological Order)",
                x_axis_label="Trade #",
                y_axis_label="Return",
                width=800,
                height=300,
                sizing_mode="stretch_width",
                toolbar_location="above",
                tools="pan,wheel_zoom,box_zoom,reset,save"
            )
            # Mark positives in one color, negatives in another
            positives = timeline_df["Return"] >= 0
            neg_source = ColumnDataSource(timeline_df[~positives])
            pos_source = ColumnDataSource(timeline_df[positives])

            p_winloss.circle(
                x="Index", y="Return",
                size=8, color="green", alpha=0.6,
                source=pos_source
            )
            p_winloss.circle(
                x="Index", y="Return",
                size=8, color="red", alpha=0.6,
                source=neg_source
            )

            # Add a hover tool (optional)
            timeline_hover = HoverTool(
                tooltips=[
                    ("Trade #", "@Index"),
                    ("Return", "@Return{0.2f}")
                ],
                mode="vline"
            )
            p_winloss.add_tools(timeline_hover)

            figures_list.append(p_winloss)
        else:
            print("No closed trades to build timeline.")
    elif show_winloss:
        print("No best trades found to build a win/loss timeline.")

    # ------------------------------------------------
    # 7) Parameter Heatmap (Optional)
    # ------------------------------------------------
    if show_param_heatmap and param_results is not None and not param_results.empty:
        # Expect param_results to have columns: e.g. ["a", "b", "final_return"]
        # We'll do a simple 2D heatmap, assuming "a" on x, "b" on y, color = "final_return"
        # Make sure "a" and "b" are numeric or categorical
        x_name = "a"
        y_name = "b"
        z_name = "final_return"
        if x_name not in param_results.columns or y_name not in param_results.columns or z_name not in param_results.columns:
            print("Param heatmap: missing expected columns in param_results. Skipping.")
        else:
            # Bokeh's categories want them as strings if discrete
            # but let's assume they're numeric for an SMA grid
            param_source = ColumnDataSource(param_results)
            # Color mapper
            low_ = param_results[z_name].min()
            high_ = param_results[z_name].max()
            mapper = LinearColorMapper(palette="Viridis256", low=low_, high=high_)

            # Convert to discrete if you want integer steps
            # or just let it float. We can skip x_range, y_range if numeric.
            p_heat = figure(
                title="Parameter Heatmap",
                width=400,
                height=300,
                toolbar_location="above",
                tools="pan,wheel_zoom,box_zoom,reset,save"
            )
            # Use rect glyph
            p_heat.rect(
                x=x_name,
                y=y_name,
                width=1,
                height=1,
                source=param_source,
                fill_color=transform(z_name, mapper),
                line_color=None
            )
            # Add color bar
            color_bar = ColorBar(color_mapper=mapper, label_standoff=8, width=8, location=(0, 0))
            p_heat.add_layout(color_bar, "right")

            # Optional hover
            heat_hover = HoverTool(
                tooltips=[
                    (f"{x_name}", f"@{x_name}"),
                    (f"{y_name}", f"@{y_name}"),
                    (f"{z_name}", f"@{z_name}{{0.2%}}"),
                ],
                mode="mouse"
            )
            p_heat.add_tools(heat_hover)

            p_heat.xaxis.axis_label = x_name
            p_heat.yaxis.axis_label = y_name
            figures_list.append(p_heat)
    elif show_param_heatmap:
        print("No valid param_results found to build heatmap (or it's empty).")

    # --------------------
    # Combine all figures
    # --------------------
    # We can do them in a single column or arrange them differently.
    layout_ = column(*figures_list, sizing_mode="stretch_width")
    show(layout_)
