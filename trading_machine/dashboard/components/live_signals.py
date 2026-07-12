"""
dashboard/components/live_signals.py — Live trading signals panel.

Displays current signals for all tickers, ranked by confidence.
Auto-refreshes every second during market hours.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.config import get_config


def render_live_signals():
    """Render the live signals panel."""
    st.header("🔴 Live Signals")

    config = get_config()

    st.info(
        "Live signals require trained models and a running data feed. "
        "Start with `python run.py live` to begin live trading mode."
    )

    # Auto-refresh
    st.markdown(f"⏱️ Auto-refresh: every {config.POLLING_INTERVAL_SECONDS}s during market hours")

    # Placeholder for signals table
    signal_placeholder = st.empty()

    # Try to load any existing signals
    try:
        from live.screener import Screener
        from models.ticker_manager import TickerManager
        from data.fetcher import FMPDataFetcher

        fetcher = FMPDataFetcher(config.get_api_key())
        ticker_mgr = TickerManager(config)
        ticker_mgr.load_all_models()

        screener = Screener(ticker_manager=ticker_mgr, fetcher=fetcher)
        screener.initialize_windows()
        signals = screener.screen_all()

        if signals:
            with signal_placeholder.container():
                st.subheader(f"📡 {len(signals)} Active Signals — {datetime.now().strftime('%H:%M:%S')}")

                # Build signal table
                signal_data = []
                for s in signals:
                    signal_data.append({
                        "Ticker": s["ticker"],
                        "Signal": s["signal_label"],
                        "Confidence": f"{s['confidence']:.1%}",
                        "Price": f"${s['price']:.2f}",
                        "Latent Norm": f"{s['latent_norm']:.4f}",
                    })

                df = pd.DataFrame(signal_data)

                # Color-code signals
                def color_signal(val):
                    if val == "LONG":
                        return "background-color: #d4edda; color: #155724"
                    elif val == "SHORT":
                        return "background-color: #f8d7da; color: #721c24"
                    return ""

                styled = df.style.applymap(color_signal, subset=["Signal"])
                st.dataframe(styled, use_container_width=True, hide_index=True)

                # Confidence bar chart
                fig = go.Figure()
                colors = [
                    "#28a745" if s["signal_label"] == "LONG"
                    else "#dc3545" if s["signal_label"] == "SHORT"
                    else "#6c757d"
                    for s in signals
                ]
                fig.add_trace(go.Bar(
                    x=[s["ticker"] for s in signals],
                    y=[s["confidence"] for s in signals],
                    marker_color=colors,
                    text=[f"{s['confidence']:.1%}" for s in signals],
                    textposition="auto",
                ))
                fig.update_layout(
                    title="Signal Confidence by Ticker",
                    xaxis_title="Ticker",
                    yaxis_title="Confidence",
                    yaxis=dict(range=[0, 1]),
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            with signal_placeholder.container():
                st.warning("No signals available. Ensure models are trained and data is loaded.")

    except Exception as e:
        st.warning(f"Live signal display unavailable: {e}")
        st.markdown("### Manual Signal Check")
        ticker_input = st.text_input("Enter ticker to check:", "SPY")
        if st.button("Check Signal") and ticker_input:
            st.info(f"Signal check for {ticker_input} is only available when the full system is running.")
