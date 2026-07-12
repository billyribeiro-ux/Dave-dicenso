"""
dashboard/components/backtest_panel.py — Backtest results viewer and runner.

Displays historical backtest performance, allows running new backtests,
and shows detailed trade-level results.
"""

from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from data.storage import DataStorage
from utils.config import get_config


def render_backtest_panel():
    """Render the backtest panel."""
    st.header("📈 Backtest Results")

    config = get_config()
    storage = DataStorage(config.DATA_DIR)

    # Load backtest results
    results = storage.load_backtest_results(limit=50)

    if len(results) == 0:
        st.info("No backtest results yet. Run backtests with:")
        st.code("python run.py backtest")
        return

    # Summary metrics
    st.subheader("Performance Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    latest = results.iloc[0]
    with col1:
        st.metric("Total Trades", int(latest["total_trades"]))
    with col2:
        st.metric("Win Rate", f"{latest['win_rate']*100:.1f}%")
    with col3:
        st.metric("Net P&L", f"${latest['net_profit']:,.2f}")
    with col4:
        st.metric("Profit Factor", f"{latest['profit_factor']:.2f}")
    with col5:
        st.metric("Max Drawdown", f"{latest['max_drawdown_pct']*100:.1f}%")

    # Additional metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sharpe Ratio", f"{latest['sharpe_ratio']:.2f}")
    with col2:
        st.metric("Avg Win", f"${latest['avg_win']:,.2f}")
    with col3:
        st.metric("Avg Loss", f"${latest['avg_loss']:,.2f}")
    with col4:
        st.metric("Largest Win", f"${latest['largest_win']:,.2f}")

    # Backtest comparison chart
    st.subheader("Backtest History")
    if len(results) > 1:
        results_sorted = results.sort_values("run_date")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_sorted["run_date"],
            y=results_sorted["win_rate"] * 100,
            mode="lines+markers",
            name="Win Rate %",
            yaxis="y1",
        ))
        fig.add_trace(go.Scatter(
            x=results_sorted["run_date"],
            y=results_sorted["sharpe_ratio"],
            mode="lines+markers",
            name="Sharpe Ratio",
            yaxis="y2",
        ))

        fig.update_layout(
            title="Backtest Performance Over Time",
            xaxis_title="Date",
            yaxis=dict(title="Win Rate (%)", side="left"),
            yaxis2=dict(title="Sharpe Ratio", side="right", overlaying="y"),
            height=400,
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Backtest results table
    st.subheader("All Backtest Runs")
    display_cols = [
        "ticker", "run_date", "total_trades", "win_rate", "net_profit",
        "profit_factor", "sharpe_ratio", "max_drawdown_pct", "model_version",
    ]
    display = results[display_cols].copy()
    display["win_rate"] = (display["win_rate"] * 100).round(1)
    display["net_profit"] = display["net_profit"].round(2)
    display["max_drawdown_pct"] = (display["max_drawdown_pct"] * 100).round(1)
    display["sharpe_ratio"] = display["sharpe_ratio"].round(2)
    display["profit_factor"] = display["profit_factor"].round(2)

    st.dataframe(display, use_container_width=True, hide_index=True)

    # Run new backtest
    st.subheader("Run New Backtest")
    st.info("Backtests are run from the command line for performance reasons.")
    st.code(f"python run.py backtest --ticker SPY")
