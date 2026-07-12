"""
dashboard/components/learning_log.py — Learning log and forensics panel.

Displays loss forensics reports, model versions, auto-adaptation history,
and the machine's self-discovered insights.
"""

import json

import pandas as pd
import streamlit as st

from data.storage import DataStorage
from utils.config import get_config


def render_learning_log():
    """Render the learning log panel showing the machine's self-discovered insights."""
    st.header("🧠 Learning Log")

    config = get_config()
    storage = DataStorage(config.DATA_DIR)

    tab1, tab2, tab3 = st.tabs(["Loss Forensics", "Model Versions", "Discoveries"])

    with tab1:
        _render_forensics(storage)

    with tab2:
        _render_model_versions(storage)

    with tab3:
        _render_discoveries(storage)


def _render_forensics(storage):
    """Render loss forensics history."""
    st.subheader("Loss Forensics Reports")

    try:
        forensics = storage.load_loss_forensics(limit=20)

        if len(forensics) == 0:
            st.info("No forensics reports yet. Loss forensics runs automatically after backtests.")
            return

        # Summary
        latest = forensics.iloc[0]
        st.caption(f"Latest report: {latest['date']} — Ticker: {latest['ticker']}")

        # Classification pie chart
        try:
            counts = json.loads(latest["classification_counts"])
            if counts:
                import plotly.graph_objects as go
                fig = go.Figure(data=[go.Pie(
                    labels=list(counts.keys()),
                    values=list(counts.values()),
                    hole=0.4,
                    marker_colors=["#ff6b6b", "#ffa502", "#3742fa", "#747d8c"],
                )])
                fig.update_layout(
                    title="Loss Classification Breakdown",
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)
        except (json.JSONDecodeError, KeyError):
            pass

        # Recommendations
        try:
            recs = json.loads(latest["recommendations"])
            if recs:
                st.subheader("Recommended Adjustments")
                for rec in recs:
                    rec_type = rec.get("type", "?")
                    action = rec.get("action", "No action specified")
                    reason = rec.get("reason", "")

                    icon = {"TYPE_A": "🎯", "TYPE_B": "🔄", "TYPE_C": "🌊", "TYPE_D": "🎲"}.get(rec_type, "📌")
                    st.markdown(f"{icon} **[{rec_type}]** {action}")
                    if reason:
                        st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;{reason}")
        except (json.JSONDecodeError, KeyError):
            pass

        # History table
        st.subheader("Forensics History")
        display = forensics[["ticker", "date"]].copy()
        st.dataframe(display, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error loading forensics: {e}")


def _render_model_versions(storage):
    """Render model version history."""
    st.subheader("Model Versions")

    config = get_config()

    try:
        versions = {}
        for ticker in config.TICKERS:
            v = storage.get_latest_model_version(ticker)
            versions[ticker] = v

        if all(v == "0.0.0" for v in versions.values()):
            st.info("No models trained yet. Train models with:")
            st.code("python run.py train")
            return

        df = pd.DataFrame(
            [(t, v) for t, v in versions.items()],
            columns=["Ticker", "Version"],
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Version display
        st.subheader("Model Status")
        for ticker, version in versions.items():
            if version != "0.0.0":
                st.success(f"✅ {ticker}: v{version}")
            else:
                st.warning(f"⚠️ {ticker}: Not trained")

    except Exception as e:
        st.error(f"Error loading model versions: {e}")


def _render_discoveries(storage):
    """Render what the machine has discovered on its own."""
    st.subheader("🪐 Self-Discovered Insights")

    st.markdown("""
    These are patterns the trading machine has discovered **entirely on its own**
    from raw price data — no human indicators were used.

    The machine learns through:
    1. **Unsupervised compression** — the world model finds structure in price sequences
    2. **Reinforcement learning** — the agent discovers profitable action patterns
    3. **Loss forensics** — losing trades are analyzed and parameters auto-adapted
    """)

    # Try to load forensics for insights
    try:
        forensics = storage.load_loss_forensics(limit=10)
        if len(forensics) > 0:
            st.subheader("Recent Forensics Insights")
            for _, row in forensics.iterrows():
                try:
                    recs = json.loads(row["recommendations"])
                    for rec in recs:
                        st.markdown(
                            f"📊 **{row['ticker']}** ({str(row['date'])[:10]}): "
                            f"[{rec.get('type', '?')}] {rec.get('action', '')}"
                        )
                except:
                    pass

    except Exception:
        pass

    # Placeholder for latent space visualization
    st.subheader("Latent Space Summary")
    st.info(
        "Latent space visualizations are available in the Charts panel. "
        "The 256-dim latent space encodes all discovered market structure "
        "without any human-defined concepts."
    )
