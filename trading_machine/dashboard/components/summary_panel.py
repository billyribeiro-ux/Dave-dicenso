"""
dashboard/components/summary_panel.py — Summary overview panel.

Shows equity curve, P&L stats, drawdown, win rate, and overall health.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.storage import DataStorage
from utils.config import get_config


def render_summary_panel():
    """Render the summary/overview panel."""
    st.header("📋 System Overview")

    config = get_config()

    col1, col2, col3, col4 = st.columns(4)

    # Try to load data for summary
    try:
        storage = DataStorage(config.DATA_DIR)

        # Load backtest results
        bt_results = storage.load_backtest_results(limit=50)
        trades = storage.load_trades(limit=500)

        if len(bt_results) == 0 and len(trades) == 0:
            st.info("No backtest results or trades yet. Train models and run backtests first.")
            st.markdown("### Quick Start")
            st.code("python run.py fetch    # Fetch historical data")
            st.code("python run.py train    # Train world models and RL agents")
            st.code("python run.py backtest # Run backtests")
            return

        with col1:
            total_trades = len(trades) if len(trades) > 0 else bt_results["total_trades"].sum()
            st.metric("Total Trades", int(total_trades))

        with col2:
            if len(trades) > 0:
                win_rate = (trades["pnl"] > 0).mean() * 100
            elif len(bt_results) > 0:
                win_rate = bt_results["win_rate"].iloc[0] * 100
            else:
                win_rate = 0
            st.metric("Win Rate", f"{win_rate:.1f}%")

        with col3:
            if len(trades) > 0:
                total_pnl = trades["pnl"].sum()
            elif len(bt_results) > 0:
                total_pnl = bt_results["net_profit"].iloc[0]
            else:
                total_pnl = 0
            st.metric("Net P&L", f"${total_pnl:,.2f}")

        with col4:
            if len(bt_results) > 0:
                sharpe = bt_results["sharpe_ratio"].iloc[0]
            else:
                sharpe = 0
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")

        # Equity curve
        st.subheader("Equity Curve")
        if len(trades) > 0:
            trades_sorted = trades.sort_values("entry_time") if "entry_time" in trades.columns else trades
            cumulative_pnl = trades_sorted["pnl"].cumsum()
            equity = config.INITIAL_CAPITAL + cumulative_pnl

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=equity.values,
                mode="lines",
                name="Equity",
                line=dict(color="blue", width=2),
            ))
            fig.add_hline(
                y=config.INITIAL_CAPITAL,
                line_dash="dash",
                line_color="gray",
                annotation_text="Initial Capital",
            )
            fig.update_layout(
                title="Equity Curve",
                xaxis_title="Trade #",
                yaxis_title="Equity ($)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Recent trades table
        st.subheader("Recent Trades")
        if len(trades) > 0:
            display_trades = trades.tail(20)[[
                "ticker", "direction", "entry_price", "exit_price", "pnl", "pnl_pct", "exit_reason"
            ]].copy()
            display_trades["pnl"] = display_trades["pnl"].round(2)
            display_trades["pnl_pct"] = (display_trades["pnl_pct"] * 100).round(2)
            display_trades = display_trades.sort_index(ascending=False)
            st.dataframe(display_trades, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading summary data: {e}")
