"""
dashboard/app.py — Streamlit dashboard main application.

Provides 5 panels:
- Summary Panel: equity curve, P&L, drawdown, win rate
- Live Signals: current ticker signals ranked by confidence
- Backtest Panel: run and view backtest results
- Learning Log: forensics reports, model versions
- Charts: Plotly OHLC + latent space visualization
"""

import os
import sys

import streamlit as st

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import get_config
from utils.logger import setup_logger

# Configure page
st.set_page_config(
    page_title="Trading Machine Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    setup_logger()
    config = get_config()

    st.title("🤖 Autonomous Trading Machine")
    st.caption("Self-learning from raw price data — no human indicators.")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Panel",
        ["Summary", "Live Signals", "Backtest", "Learning Log", "Charts"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Tickers:** {', '.join(config.TICKERS[:5])}...")
    st.sidebar.markdown(f"**Model Dir:** `{config.MODEL_DIR}`")
    st.sidebar.markdown(f"**Data Dir:** `{config.DATA_DIR}`")
    st.sidebar.markdown(f"**Port:** {config.DASHBOARD_PORT}")

    # Route to panels
    try:
        if page == "Summary":
            from dashboard.components.summary_panel import render_summary_panel
            render_summary_panel()
        elif page == "Live Signals":
            from dashboard.components.live_signals import render_live_signals
            render_live_signals()
        elif page == "Backtest":
            from dashboard.components.backtest_panel import render_backtest_panel
            render_backtest_panel()
        elif page == "Learning Log":
            from dashboard.components.learning_log import render_learning_log
            render_learning_log()
        elif page == "Charts":
            from dashboard.components.charts import render_charts
            render_charts()
    except ImportError as e:
        st.error(f"Failed to load component: {e}")
        st.info("Some components require trained models or data. Train models first using `python run.py train`.")
    except Exception as e:
        st.error(f"Error rendering panel: {e}")


if __name__ == "__main__":
    main()
