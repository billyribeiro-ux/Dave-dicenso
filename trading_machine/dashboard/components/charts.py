"""
dashboard/components/charts.py — Price charts and latent space visualization.

Shows OHLCV price data with Plotly and latent space PCA/t-SNE
visualizations of the world model's internal representations.
"""

from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.storage import DataStorage, DataNotFoundError
from utils.config import get_config


def render_charts():
    """Render the charts panel."""
    st.header("📊 Charts & Visualizations")

    config = get_config()

    tab1, tab2, tab3 = st.tabs(["Price Chart", "Latent Space", "Regime Map"])

    with tab1:
        _render_price_chart(config)

    with tab2:
        _render_latent_space(config)

    with tab3:
        _render_regime_map(config)


def _render_price_chart(config):
    """Render interactive OHLCV price chart."""
    st.subheader("Price Chart")

    ticker = st.selectbox("Select Ticker", config.TICKERS, key="chart_ticker")
    days_back = st.slider("Days to display", 1, 100, 30, key="chart_days")

    try:
        storage = DataStorage(config.DATA_DIR)
        df = storage.load_tick_data(ticker)

        if len(df) == 0:
            st.warning(f"No data for {ticker}")
            return

        # Filter to last N days
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            cutoff = df["timestamp"].max() - pd.Timedelta(days=days_back)
            df = df[df["timestamp"] >= cutoff]

        if len(df) == 0:
            st.warning(f"No data in the last {days_back} days for {ticker}")
            return

        # Check if we have OHLC data (intraday) or just close
        has_ohlc = all(c in df.columns for c in ["open", "high", "low", "close"])

        if has_ohlc:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
            )

            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df["timestamp"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    name=ticker,
                ),
                row=1, col=1,
            )

            # Volume bars
            if "volume" in df.columns:
                colors = ["green" if c >= o else "red" for c, o in zip(df["close"], df["open"])]
                fig.add_trace(
                    go.Bar(x=df["timestamp"], y=df["volume"], name="Volume", marker_color=colors),
                    row=2, col=1,
                )

            fig.update_layout(
                title=f"{ticker} — Last {days_back} Days",
                xaxis_rangeslider_visible=False,
                height=600,
            )
            fig.update_yaxis(title_text="Price ($)", row=1, col=1)
            fig.update_yaxis(title_text="Volume", row=2, col=1)

        else:
            # Line chart for close-only data
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["timestamp"],
                y=df["close"],
                mode="lines",
                name=f"{ticker} Close",
                line=dict(color="blue", width=1),
            ))
            fig.update_layout(
                title=f"{ticker} Close Price — Last {days_back} Days",
                xaxis_title="Time",
                yaxis_title="Price ($)",
                height=500,
            )

        st.plotly_chart(fig, use_container_width=True)

        # Price statistics
        if "close" in df.columns:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${df['close'].iloc[-1]:.2f}")
            with col2:
                st.metric("Day High", f"${df['high'].max():.2f}" if has_ohlc else f"${df['close'].max():.2f}")
            with col3:
                st.metric("Day Low", f"${df['low'].min():.2f}" if has_ohlc else f"${df['close'].min():.2f}")
            with col4:
                returns = df["close"].pct_change().dropna()
                volatility = returns.std() * np.sqrt(390) * 100  # Annualized
                st.metric("Volatility (ann.)", f"{volatility:.1f}%")

    except DataNotFoundError:
        st.warning(f"No data stored for {ticker}. Fetch data first.")
    except Exception as e:
        st.error(f"Error rendering chart: {e}")


def _render_latent_space(config):
    """Render latent space visualization using PCA."""
    st.subheader("Latent Space Visualization")

    ticker = st.selectbox("Select Ticker", config.TICKERS, key="latent_ticker")

    st.info(
        "Latent space visualization requires a trained world model. "
        "Each point is a 256-dim latent vector projected to 2D via PCA. "
        "The machine discovers structure in price data without any human concepts."
    )

    try:
        from models.world_model import PriceVAE
        import torch
        from sklearn.decomposition import PCA

        # Load data
        storage = DataStorage(config.DATA_DIR)
        prices = storage.get_close_prices(ticker)

        if len(prices) < 600:
            st.warning(f"Not enough data for {ticker} ({len(prices)} prices, need 600+)")
            return

        # Try to load the world model
        model_dir = f"{config.MODEL_DIR}/{ticker}"
        import os
        wm_files = sorted([
            f for f in os.listdir(model_dir)
            if f.startswith("world_model_v") and f.endswith(".pt")
        ]) if os.path.isdir(model_dir) else []

        if not wm_files:
            st.warning(f"No trained world model found for {ticker}. Train models first.")
            return

        # Load model
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = PriceVAE(
            input_dim=1,
            latent_dim=config.LATENT_DIM,
            sequence_length=config.INPUT_WINDOW_TICKS,
            future_length=config.FUTURE_WINDOW_TICKS,
        ).to(device)
        model.load_state_dict(torch.load(
            os.path.join(model_dir, wm_files[-1]),
            map_location=device,
        ))
        model.eval()

        # Encode rolling windows
        window_size = config.INPUT_WINDOW_TICKS
        stride = 100
        latent_vectors = []
        timestamps = []

        for i in range(0, len(prices) - window_size, stride):
            window = prices[i:i + window_size].astype(np.float32)
            mean = window.mean()
            std = window.std()
            if std < 1e-8:
                std = 1.0
            normalized = (window - mean) / std

            x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            with torch.no_grad():
                mu, logvar, z = model.encode(x)
            latent_vectors.append(z.cpu().numpy().flatten())
            timestamps.append(i)

        if len(latent_vectors) < 2:
            st.warning("Not enough windows to visualize latent space")
            return

        # PCA to 2D
        latent_matrix = np.stack(latent_vectors)
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_matrix)

        # Color by time (earlier = blue, later = red)
        colors = np.linspace(0, 1, len(latent_2d))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=latent_2d[:, 0],
            y=latent_2d[:, 1],
            mode="lines+markers",
            marker=dict(
                size=5,
                color=colors,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Time →"),
            ),
            line=dict(width=1, color="gray"),
            name="Latent Trajectory",
        ))

        fig.update_layout(
            title=f"{ticker} Latent Space Trajectory (PCA, {pca.explained_variance_ratio_.sum()*100:.0f}% variance)",
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "The machine discovers this structure entirely from closing prices. "
            "No RSI, MACD, Bollinger Bands, or any human-invented indicator was used. "
            "Each trajectory segment represents a different market 'regime' — "
            "trending, ranging, volatile, calm — discovered autonomously."
        )

    except Exception as e:
        st.error(f"Error rendering latent space: {e}")


def _render_regime_map(config):
    """Render regime map from latent space clusters."""
    st.subheader("Regime Detection Map")

    ticker = st.selectbox("Select Ticker", config.TICKERS, key="regime_ticker")

    st.info(
        "Regime boundaries are detected by measuring cosine distance "
        "between consecutive latent vectors. When the distance exceeds "
        "a threshold (0.5), a new regime is flagged — no volatility "
        "or trend indicators needed."
    )

    try:
        from evolution.regime_detector import RegimeDetector

        storage = DataStorage(config.DATA_DIR)
        prices = storage.get_close_prices(ticker)

        if len(prices) < 600:
            st.warning(f"Not enough data for {ticker}")
            return

        # Load world model
        model_dir = f"{config.MODEL_DIR}/{ticker}"
        import os
        wm_files = sorted([
            f for f in os.listdir(model_dir)
            if f.startswith("world_model_v") and f.endswith(".pt")
        ]) if os.path.isdir(model_dir) else []

        if not wm_files:
            st.warning(f"No trained world model for {ticker}")
            return

        import torch
        from models.world_model import PriceVAE

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = PriceVAE(
            input_dim=1, latent_dim=config.LATENT_DIM,
            sequence_length=config.INPUT_WINDOW_TICKS,
            future_length=config.FUTURE_WINDOW_TICKS,
        ).to(device)
        model.load_state_dict(torch.load(
            os.path.join(model_dir, wm_files[-1]), map_location=device
        ))
        model.eval()

        detector = RegimeDetector(world_model=model, threshold=0.5)
        boundaries = detector.detect_regimes(prices, window_size=500, stride=50)

        if not boundaries:
            st.info("No regime boundaries detected. The market has been stable.")
            return

        stats = detector.get_regime_statistics()

        st.metric("Regimes Detected", stats["num_regimes"])
        st.metric("Avg Regime Length", f"{stats['avg_regime_length']:.0f} ticks")
        st.metric("Boundary Count", stats["total_boundaries"])

        # Visualize regimes on price chart
        window_size = 500
        fig = go.Figure()

        # Price line
        price_indices = np.arange(len(prices))
        fig.add_trace(go.Scatter(
            x=price_indices,
            y=prices,
            mode="lines",
            name="Price",
            line=dict(color="gray", width=0.5),
        ))

        # Regime boundary markers
        for b in boundaries:
            fig.add_vline(
                x=b["index"],
                line_dash="dash",
                line_color="red",
                line_width=1,
                opacity=0.5,
            )

        fig.update_layout(
            title=f"{ticker} — Regime Boundaries (cosine distance > 0.5)",
            xaxis_title="Tick Index",
            yaxis_title="Price ($)",
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Red dashed lines mark points where the world model's latent representation "
            "shifted significantly — indicating a market regime change. "
            "This is discovered purely from price structure, not from any indicator."
        )

    except Exception as e:
        st.error(f"Error rendering regime map: {e}")
