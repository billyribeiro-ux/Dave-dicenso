"""
data/fetcher.py — FMP (Financial Modeling Prep) API data fetcher.
Implements rate limiting, exponential backoff, caching, and batch quoting.
"""

import time as _time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import pytz
import requests
from tqdm import tqdm

from data.storage import DataStorage
from utils.config import get_config
from utils.logger import get_logger


class DataFetchError(Exception):
    """Custom exception raised when data fetching fails unrecoverably."""
    pass


class FMPDataFetcher:
    """Fetches financial data from Financial Modeling Prep API.

    Features:
    - Rate limiting: max 300 requests/minute (tracked via deque of timestamps)
    - Exponential backoff on 429: 1s, 2s, 4s, 8s, 16s, then raise
    - Quote caching with 60-second TTL
    - Batch quote endpoint for efficiency
    - Retries up to 3x on connection errors
    """

    BASE_URL = "https://financialmodelingprep.com/api/v3"
    MAX_REQUESTS_PER_MINUTE = 300
    CACHE_TTL_SECONDS = 60
    MAX_RETRIES = 3

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._logger = get_logger()
        self._request_timestamps: deque = deque()
        self._quote_cache: Dict[str, Dict] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._est_tz = pytz.timezone("US/Eastern")

    # ------------------------------------------------------------------
    # Rate limiting & error handling
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Enforce maximum 300 requests per minute.

        Tracks request timestamps in a deque. If we've made 300 requests
        in the last 60 seconds, sleep until the oldest falls outside the window.
        """
        now = _time.monotonic()
        cutoff = now - 60.0

        # Remove timestamps older than 60 seconds
        while self._request_timestamps and self._request_timestamps[0] < cutoff:
            self._request_timestamps.popleft()

        if len(self._request_timestamps) >= self.MAX_REQUESTS_PER_MINUTE:
            wait = self._request_timestamps[0] - cutoff + 0.1
            if wait > 0:
                _time.sleep(wait)
            # Re-clean after sleep
            cutoff = _time.monotonic() - 60.0
            while self._request_timestamps and self._request_timestamps[0] < cutoff:
                self._request_timestamps.popleft()

        self._request_timestamps.append(_time.monotonic())

    def _make_request(self, url: str, params: Optional[Dict] = None) -> requests.Response:
        """Make an HTTP GET request with rate limiting + retries + exponential backoff.

        Args:
            url: Full URL or endpoint path.
            params: Optional query parameters.

        Returns:
            requests.Response object.

        Raises:
            DataFetchError: After exhausting retries or on unrecoverable errors.
        """
        if params is None:
            params = {}
        params["apikey"] = self._api_key

        self._rate_limit()

        backoff = 1.0
        last_exception = None

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    self._logger.warning(f"Rate limited (429). Backing off {backoff}s")
                    _time.sleep(backoff)
                    backoff = min(backoff * 2, 16.0)
                    if backoff > 16.0:
                        self._rate_limit()
                        resp = requests.get(url, params=params, timeout=30)
                        if resp.status_code == 429:
                            raise DataFetchError("Rate limited after max backoff")
                        return resp
                    continue

                if resp.status_code >= 500:
                    self._logger.warning(f"Server error {resp.status_code}, attempt {attempt+1}")
                    if attempt < self.MAX_RETRIES:
                        _time.sleep(backoff)
                        backoff = min(backoff * 2, 8.0)
                        continue
                    raise DataFetchError(f"Server error {resp.status_code}: {resp.text[:200]}")

                if resp.status_code >= 400:
                    raise DataFetchError(f"HTTP {resp.status_code}: {resp.text[:500]}")

                return resp

            except requests.exceptions.ConnectionError as e:
                last_exception = e
                self._logger.warning(f"Connection error (attempt {attempt+1}): {e}")
                if attempt < self.MAX_RETRIES:
                    _time.sleep(backoff)
                    backoff = min(backoff * 2, 8.0)
                else:
                    raise DataFetchError(f"Connection failed after {self.MAX_RETRIES} retries") from e

            except requests.exceptions.Timeout as e:
                last_exception = e
                self._logger.warning(f"Timeout (attempt {attempt+1}): {e}")
                if attempt < self.MAX_RETRIES:
                    _time.sleep(backoff)
                    backoff = min(backoff * 2, 8.0)
                else:
                    raise DataFetchError(f"Timeout after {self.MAX_RETRIES} retries") from e

        raise DataFetchError(f"Request failed: {last_exception}")

    # ------------------------------------------------------------------
    # Historical data fetching
    # ------------------------------------------------------------------

    def fetch_historical_intraday(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch 1-minute intraday OHLCV data from FMP.

        Endpoint: /historical-chart/1min/{ticker}

        Returns a DataFrame with columns: timestamp, open, high, low, close, volume.
        Filters to 09:30-16:00 EST only.
        """
        url = f"{self.BASE_URL}/historical-chart/1min/{ticker}"
        params = {"from": start_date, "to": end_date}

        self._logger.info(f"Fetching intraday data for {ticker} ({start_date} to {end_date})")

        resp = self._make_request(url, params)
        data = resp.json()

        if isinstance(data, dict) and "Error Message" in data:
            raise DataFetchError(f"FMP error for {ticker}: {data['Error Message']}")

        if not isinstance(data, list):
            raise DataFetchError(f"Unexpected response format for {ticker}: {type(data)}")

        if len(data) == 0:
            self._logger.warning(f"No intraday data returned for {ticker}")
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(data)

        # Convert date strings to datetime with EST timezone
        df["timestamp"] = pd.to_datetime(df["date"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(self._est_tz)
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert(self._est_tz)

        # Filter to market hours only (09:30 - 16:00 EST)
        df = df[
            (df["timestamp"].dt.time >= pd.Timestamp("09:30").time()) &
            (df["timestamp"].dt.time <= pd.Timestamp("16:00").time())
        ]

        # Select and rename columns
        df = df.rename(columns={
            "date": "timestamp_orig",
        })
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        df = df.sort_values("timestamp").reset_index(drop=True)

        self._logger.info(f"Fetched {len(df)} intraday rows for {ticker}")
        return df

    def fetch_historical_daily(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch daily OHLCV data from FMP.

        Used as fallback when intraday data is unavailable.
        Endpoint: /historical-price-full/{ticker}
        """
        url = f"{self.BASE_URL}/historical-price-full/{ticker}"
        params = {"from": start_date, "to": end_date}

        self._logger.info(f"Fetching daily data for {ticker} ({start_date} to {end_date})")

        resp = self._make_request(url, params)
        data = resp.json()

        if isinstance(data, dict) and "Error Message" in data:
            raise DataFetchError(f"FMP error for {ticker}: {data['Error Message']}")

        if isinstance(data, dict) and "historical" in data:
            records = data["historical"]
        elif isinstance(data, list):
            records = data
        else:
            raise DataFetchError(f"Unexpected daily response format for {ticker}")

        if not records:
            self._logger.warning(f"No daily data returned for {ticker}")
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["date"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(self._est_tz)

        df = df.rename(columns={
            "date": "timestamp_orig",
        })
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        df = df.sort_values("timestamp").reset_index(drop=True)

        self._logger.info(f"Fetched {len(df)} daily rows for {ticker}")
        return df

    # ------------------------------------------------------------------
    # Real-time quote fetching
    # ------------------------------------------------------------------

    def _cache_quote(self, ticker: str, quote: dict) -> None:
        """Store quote in cache with timestamp."""
        self._quote_cache[ticker] = quote
        self._cache_timestamps[ticker] = _time.monotonic()

    def _get_cached_quote(self, ticker: str) -> Optional[dict]:
        """Return cached quote if it's less than CACHE_TTL_SECONDS old."""
        if ticker not in self._quote_cache:
            return None
        age = _time.monotonic() - self._cache_timestamps.get(ticker, 0)
        if age > self.CACHE_TTL_SECONDS:
            return None
        return self._quote_cache[ticker]

    def fetch_real_time_quote(self, ticker: str) -> dict:
        """Fetch real-time quote for a single ticker.

        Must execute in under 200ms to maintain 1-second polling cycle.
        Uses caching: returns cached value if <60s old; raises on failure if cache is stale.

        Returns dict with keys: symbol, price, change, changePercentage, timestamp.
        """
        cached = self._get_cached_quote(ticker)
        if cached is not None:
            return cached

        url = f"{self.BASE_URL}/quote/{ticker}"

        try:
            resp = self._make_request(url)
            data = resp.json()

            if isinstance(data, list) and len(data) > 0:
                quote = data[0]
            elif isinstance(data, dict):
                quote = data
            else:
                raise DataFetchError(f"Unexpected quote format for {ticker}")

            # Standardize keys
            result = {
                "symbol": quote.get("symbol", ticker),
                "price": float(quote.get("price", 0)),
                "change": float(quote.get("change", 0)),
                "changePercentage": float(quote.get("changesPercentage", 0)),
                "timestamp": datetime.now(self._est_tz),
            }
            self._cache_quote(ticker, result)
            return result

        except DataFetchError:
            # If we have ANY cached value, return it regardless of age on failure
            if ticker in self._quote_cache:
                self._logger.warning(f"Using stale cache for {ticker} after fetch failure")
                return self._quote_cache[ticker]
            raise

    def fetch_batch_quotes(self, tickers: List[str]) -> Dict[str, dict]:
        """Fetch quotes for multiple tickers in a single request.

        More efficient than calling fetch_real_time_quote for each ticker.
        Returns dict mapping ticker -> quote data.
        """
        if not tickers:
            return {}

        # Check cache first
        result = {}
        uncached = []
        for t in tickers:
            cached = self._get_cached_quote(t)
            if cached is not None:
                result[t] = cached
            else:
                uncached.append(t)

        if not uncached:
            return result

        ticker_str = ",".join(uncached)
        url = f"{self.BASE_URL}/quote/{ticker_str}"

        try:
            resp = self._make_request(url)
            data = resp.json()

            if not isinstance(data, list):
                # On failure, fill remaining with cache or skip
                for t in uncached:
                    if t in self._quote_cache:
                        result[t] = self._quote_cache[t]
                return result

            for quote in data:
                symbol = quote.get("symbol", "")
                q = {
                    "symbol": symbol,
                    "price": float(quote.get("price", 0)),
                    "change": float(quote.get("change", 0)),
                    "changePercentage": float(quote.get("changesPercentage", 0)),
                    "timestamp": datetime.now(self._est_tz),
                }
                self._cache_quote(symbol, q)
                result[symbol] = q

            # Fill any tickers not returned by the batch call
            for t in uncached:
                if t not in result and t in self._quote_cache:
                    result[t] = self._quote_cache[t]

            return result

        except DataFetchError:
            for t in uncached:
                if t in self._quote_cache:
                    result[t] = self._quote_cache[t]
            return result

    # ------------------------------------------------------------------
    # Bulk historical fetching
    # ------------------------------------------------------------------

    def fetch_all_historical_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch historical intraday data for all tickers in the config.

        Stores results using DataStorage. Shows tqdm progress bar.
        Handles failures gracefully (one ticker failure doesn't stop others).
        Logs every fetch attempt.
        """
        config = get_config()
        storage = DataStorage(config.DATA_DIR)
        results: Dict[str, pd.DataFrame] = {}

        self._logger.info(f"Starting bulk fetch for {len(config.TICKERS)} tickers")

        for ticker in tqdm(config.TICKERS, desc="Fetching tickers"):
            self._logger.info(f"Fetching data for {ticker}")
            try:
                df = self.fetch_historical_intraday(
                    ticker, config.START_DATE, config.END_DATE
                )
                if len(df) > 0:
                    storage.save_tick_data(ticker, df)
                    results[ticker] = df
                    self._logger.info(
                        f"Saved {len(df)} rows for {ticker}"
                    )
                else:
                    # Try daily as fallback
                    self._logger.warning(
                        f"No intraday data for {ticker}, trying daily"
                    )
                    df = self.fetch_historical_daily(
                        ticker, config.START_DATE, config.END_DATE
                    )
                    if len(df) > 0:
                        storage.save_tick_data(ticker, df)
                        results[ticker] = df
                        self._logger.info(
                            f"Saved {len(df)} daily rows for {ticker}"
                        )
                    else:
                        self._logger.error(f"No data at all for {ticker}, skipping")
            except DataFetchError as e:
                self._logger.error(f"Failed to fetch {ticker}: {e}")
                continue
            except Exception as e:
                self._logger.error(f"Unexpected error fetching {ticker}: {e}")
                continue

        self._logger.info(f"Bulk fetch complete. Got data for {len(results)}/{len(config.TICKERS)} tickers")
        return results
