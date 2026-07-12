"""
data/validation.py — Data quality validation for tick data and realtime quotes.
Checks completeness, handles nulls, detects suspicious price movements.
"""

from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger


class DataValidator:
    """Validates and cleans financial tick data and realtime quotes."""

    MARKET_OPEN = "09:30"
    MARKET_CLOSE = "16:00"
    MAX_PRICE_JUMP_PCT = 0.20  # 20% jump flag
    MAX_FORWARD_FILL = 5  # Max consecutive forward-fills before flagging
    MAX_QUOTE_AGE_SECONDS = 60
    MAX_CHANGE_PCT = 50.0  # +/-50%

    def __init__(self):
        self._logger = get_logger()

    # ------------------------------------------------------------------
    # Tick data validation
    # ------------------------------------------------------------------

    def validate_tick_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate a tick data DataFrame.

        Returns (is_valid: bool, issues: list of str).
        """
        issues: List[str] = []
        required_cols = {"timestamp", "open", "high", "low", "close", "volume"}

        # Check required columns
        missing = required_cols - set(df.columns)
        if missing:
            issues.append(f"Missing columns: {missing}")
            return False, issues

        # Check for null values in close column
        if df["close"].isnull().any():
            null_count = df["close"].isnull().sum()
            issues.append(f"Null values in close column: {null_count} rows")

        # Check timestamps are within market hours
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            open_time = pd.Timestamp(self.MARKET_OPEN).time()
            close_time = pd.Timestamp(self.MARKET_CLOSE).time()
            times = ts.dt.time
            outside = (times < open_time) | (times > close_time)
            if outside.any():
                issues.append(f"Timestamps outside market hours ({self.MARKET_OPEN}-{self.MARKET_CLOSE}): {outside.sum()} rows")

        # Check for duplicate timestamps
        if "timestamp" in df.columns:
            dupes = df["timestamp"].duplicated()
            if dupes.any():
                issues.append(f"Duplicate timestamps: {dupes.sum()} rows")

        # Check close price is not negative or zero
        if (df["close"] <= 0).any():
            bad = (df["close"] <= 0).sum()
            issues.append(f"Non-positive close prices: {bad} rows")

        # Check for price jumps >20% between consecutive ticks
        if len(df) > 1:
            pct_changes = df["close"].pct_change().abs()
            jumps = pct_changes > self.MAX_PRICE_JUMP_PCT
            if jumps.any():
                issues.append(f"Suspicious price jumps >{self.MAX_PRICE_JUMP_PCT*100}%: {jumps.sum()} occurrences (flagged, not removed)")

        is_valid = len(issues) == 0 or all(
            "suspicious" in i.lower() or "flagged" in i.lower()
            for i in issues
        )

        return is_valid, issues

    # ------------------------------------------------------------------
    # Tick data cleaning
    # ------------------------------------------------------------------

    def clean_tick_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean tick data:
        - Remove duplicate timestamps (keep first)
        - Forward-fill null close values (max 5 consecutive fills)
        - Remove rows outside market hours
        - Sort by timestamp ascending
        """
        df = df.copy()

        # Sort by timestamp
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

        # Remove duplicate timestamps, keep first
        if "timestamp" in df.columns:
            df = df.drop_duplicates(subset=["timestamp"], keep="first")

        # Forward-fill null close values, track consecutive fills
        if "close" in df.columns:
            null_mask = df["close"].isnull()
            if null_mask.any():
                # Count consecutive nulls
                consecutive = 0
                for i in range(len(df)):
                    if pd.isna(df["close"].iloc[i]):
                        consecutive += 1
                        if consecutive > self.MAX_FORWARD_FILL:
                            self._logger.warning(
                                f"More than {self.MAX_FORWARD_FILL} consecutive nulls at index {i}, stopping forward-fill"
                            )
                            break
                    else:
                        consecutive = 0

                df["close"] = df["close"].ffill(limit=self.MAX_FORWARD_FILL)

        # Remove rows outside market hours
        if "timestamp" in df.columns:
            open_time = pd.Timestamp(self.MARKET_OPEN).time()
            close_time = pd.Timestamp(self.MARKET_CLOSE).time()
            times = df["timestamp"].dt.time
            df = df[(times >= open_time) & (times <= close_time)]

        # Drop any remaining rows with null close
        if "close" in df.columns:
            df = df.dropna(subset=["close"])

        df = df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Data completeness check
    # ------------------------------------------------------------------

    def check_data_completeness(
        self, ticker: str, start_date: str, end_date: str
    ) -> float:
        """Count expected vs actual trading days. Returns completeness percentage.

        Excludes weekends and market holidays.
        Flags gaps > 30 minutes during a trading day as suspicious.
        """
        from utils.scheduler import US_MARKET_HOLIDAYS

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        # Count all business days (Mon-Fri)
        all_days = pd.bdate_range(start=start, end=end)

        # Remove holidays
        expected_days = 0
        for d in all_days:
            d_date = d.date()
            if d_date not in US_MARKET_HOLIDAYS:
                expected_days += 1

        if expected_days == 0:
            return 0.0

        completeness_pct = 0.0

        try:
            from data.storage import DataStorage
            from utils.config import get_config

            config = get_config()
            storage = DataStorage(config.DATA_DIR)
            df = storage.load_tick_data(ticker, start_date, end_date)

            if len(df) == 0:
                return 0.0

            # Count unique trading days with data
            actual_days = df["timestamp"].dt.date.nunique()
            completeness_pct = (actual_days / expected_days) * 100.0

            # Check for intraday gaps > 30 minutes
            if len(df) > 1:
                ts_sorted = df["timestamp"].sort_values()
                diffs = ts_sorted.diff().dropna()
                large_gaps = diffs[diffs > pd.Timedelta(minutes=30)]
                if len(large_gaps) > 0:
                    self._logger.warning(
                        f"{ticker}: {len(large_gaps)} intraday gaps >30min detected"
                    )
        except Exception as e:
            self._logger.error(f"Error checking completeness for {ticker}: {e}")

        return completeness_pct

    # ------------------------------------------------------------------
    # Realtime quote validation
    # ------------------------------------------------------------------

    def validate_realtime_quote(self, quote_dict: dict) -> Tuple[bool, List[str]]:
        """Validate a realtime quote dictionary.

        Returns (is_valid, issues).
        """
        issues: List[str] = []

        # Check price is positive
        price = quote_dict.get("price", 0)
        if price <= 0:
            issues.append(f"Non-positive price: {price}")

        # Check timestamp is recent (within 60 seconds)
        ts = quote_dict.get("timestamp")
        if ts is not None:
            age = (datetime.now() - ts).total_seconds()
            if age > self.MAX_QUOTE_AGE_SECONDS:
                issues.append(f"Stale quote: {age:.1f}s old (max {self.MAX_QUOTE_AGE_SECONDS}s)")

        # Check changePercentage within bounds
        change_pct = quote_dict.get("changePercentage", 0)
        if abs(change_pct) > self.MAX_CHANGE_PCT:
            issues.append(f"Unreasonable changePercentage: {change_pct}% (bounds: ±{self.MAX_CHANGE_PCT}%)")

        is_valid = len(issues) == 0
        return is_valid, issues
