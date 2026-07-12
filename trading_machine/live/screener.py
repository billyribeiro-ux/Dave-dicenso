"""
live/screener.py — Real-time ticker screening during market hours.

Polls all tickers every 1 second, encodes price windows through world models,
gets signals from RL agents, and ranks by confidence score.
"""

import time as _time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.fetcher import FMPDataFetcher, DataFetchError
from data.storage import DataStorage, DataNotFoundError
from data.validation import DataValidator
from utils.config import get_config
from utils.logger import get_logger


class Screener:
    """Real-time market screener that generates trading signals.

    For each ticker:
    1. Fetch latest price + quote via batch API
    2. Maintain a rolling 500-tick price window
    3. Encode window through world model → latent state
    4. Get action + confidence from RL agent
    5. Rank signals by confidence
    """

    SIGNAL_LONG = 0
    SIGNAL_SHORT = 1
    SIGNAL_NEUTRAL = 2

    def __init__(self, ticker_manager=None, fetcher=None, storage=None):
        self.ticker_manager = ticker_manager
        self.fetcher = fetcher
        self.storage = storage or DataStorage(get_config().DATA_DIR)
        self.config = get_config()
        self.logger = get_logger()
        self.validator = DataValidator()

        # Rolling price windows per ticker
        self._price_windows: Dict[str, deque] = {}
        self._window_size = self.config.INPUT_WINDOW_TICKS

        # Latest signals
        self._latest_signals: Dict[str, dict] = {}

        # Initialize windows for all tickers
        for ticker in self.config.TICKERS:
            self._price_windows[ticker] = deque(maxlen=self._window_size)

    def initialize_windows(self) -> int:
        """Pre-fill price windows with historical data for each ticker.

        Returns the number of tickers successfully initialized.
        """
        initialized = 0
        for ticker in self.config.TICKERS:
            try:
                prices = self.storage.get_close_prices(ticker)
                if len(prices) >= self._window_size:
                    recent = prices[-self._window_size:]
                    for p in recent:
                        self._price_windows[ticker].append(float(p))
                    initialized += 1
                    self.logger.info(
                        f"Initialized window for {ticker}: {len(recent)} prices"
                    )
                else:
                    self.logger.warning(
                        f"Not enough data for {ticker}: {len(prices)} prices, need {self._window_size}"
                    )
            except (DataNotFoundError, Exception) as e:
                self.logger.error(f"Failed to initialize window for {ticker}: {e}")

        return initialized

    def update_price(self, ticker: str, price: float) -> None:
        """Add a new price to the rolling window."""
        self._price_windows[ticker].append(price)

    def get_price_window(self, ticker: str) -> Optional[np.ndarray]:
        """Get the current price window as a numpy array.

        Returns None if window is not full yet.
        """
        window = self._price_windows[ticker]
        if len(window) < self._window_size:
            return None
        return np.array(list(window), dtype=np.float64)

    def screen_single(self, ticker: str) -> Optional[dict]:
        """Screen a single ticker and return its signal.

        Returns dict with: ticker, signal, confidence, price, latent_norm, timestamp.
        Returns None if ticker is not ready (window not full, models not trained).
        """
        if self.ticker_manager is None:
            self.logger.error("No ticker manager configured for screening")
            return None

        if not self.ticker_manager.is_ticker_ready(ticker):
            return None

        price_window = self.get_price_window(ticker)
        if price_window is None:
            return None

        try:
            # Get latent state
            latent = self.ticker_manager.get_latent_state(ticker, price_window)

            # Get signal
            action, confidence = self.ticker_manager.get_signal(ticker, latent)

            signal = {
                "ticker": ticker,
                "signal": action,
                "signal_label": self._signal_label(action),
                "confidence": confidence,
                "price": float(price_window[-1]),
                "latent_norm": float(np.linalg.norm(latent)),
                "timestamp": datetime.now(),
            }

            self._latest_signals[ticker] = signal
            return signal

        except Exception as e:
            self.logger.error(f"Screening error for {ticker}: {e}")
            return None

    def _signal_label(self, action: int) -> str:
        if action == self.SIGNAL_LONG:
            return "LONG"
        elif action == self.SIGNAL_SHORT:
            return "SHORT"
        else:
            return "NEUTRAL"

    def screen_all(self) -> List[dict]:
        """Screen all tickers and return ranked signals.

        Updates price windows from the fetcher if available.
        Returns list of signal dicts sorted by confidence descending.
        """
        # Update prices via batch quote if fetcher available
        if self.fetcher is not None:
            try:
                tickers = [t for t in self.config.TICKERS
                          if self.ticker_manager is not None
                          and self.ticker_manager.is_ticker_ready(t)]
                if tickers:
                    quotes = self.fetcher.fetch_batch_quotes(tickers)
                    for ticker, quote in quotes.items():
                        price = quote.get("price", 0)
                        if price > 0:
                            self.update_price(ticker, price)
            except (DataFetchError, Exception) as e:
                self.logger.warning(f"Batch quote fetch failed: {e}")

        # Screen each ticker
        signals = []
        for ticker in self.config.TICKERS:
            signal = self.screen_single(ticker)
            if signal is not None:
                signals.append(signal)

        # Sort by confidence descending, LONG/SHORT first, then NEUTRAL
        signals.sort(
            key=lambda s: (
                0 if s["signal"] != self.SIGNAL_NEUTRAL else 1,
                -s["confidence"],
            )
        )

        return signals

    def get_latest_signal(self, ticker: str) -> Optional[dict]:
        """Get the most recent signal for a ticker."""
        return self._latest_signals.get(ticker)

    def get_all_latest_signals(self) -> Dict[str, dict]:
        """Get latest signals for all tickers."""
        return dict(self._latest_signals)

    def get_top_signals(self, n: int = 5, exclude_neutral: bool = True) -> List[dict]:
        """Get top N signals by confidence."""
        signals = self.screen_all()
        if exclude_neutral:
            signals = [s for s in signals if s["signal"] != self.SIGNAL_NEUTRAL]
        return signals[:n]

    def run_polling_cycle(self) -> List[dict]:
        """Run one complete polling cycle: fetch → update → screen → rank.

        Designed to be called every POLLING_INTERVAL_SECONDS by the scheduler.
        """
        return self.screen_all()
