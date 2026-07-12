"""
live/position_manager.py — Live position tracking and risk management.

Tracks open positions, calculates unrealized P&L, enforces max drawdown,
and logs all trades to the database.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from utils.config import get_config
from utils.logger import get_logger


class PositionManager:
    """Tracks live trading positions and enforces risk limits.

    Maintains a single position per ticker. Tracks entry/exit, P&L,
    and equity curve for drawdown monitoring.
    """

    def __init__(self, storage=None):
        self.config = get_config()
        self.storage = storage
        self.logger = get_logger()

        # Active positions: ticker → position dict
        self._positions: Dict[str, dict] = {}
        self._trade_history: List[dict] = []
        self._equity_curve: List[Tuple[datetime, float]] = []

        # Account state
        self.initial_capital = self.config.INITIAL_CAPITAL
        self.cash = self.config.INITIAL_CAPITAL
        self.peak_equity = self.config.INITIAL_CAPITAL
        self.max_drawdown_pct = self.config.MAX_DRAWDOWN_PCT

    @property
    def equity(self) -> float:
        """Total equity = cash + unrealized P&L of all open positions."""
        unrealized = 0.0
        for pos in self._positions.values():
            unrealized += pos.get("unrealized_pnl", 0.0)
        return self.cash + unrealized

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown as fraction of peak equity."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity

    @property
    def is_max_drawdown_exceeded(self) -> bool:
        """Check if drawdown exceeds the configured maximum."""
        return self.drawdown_pct >= self.max_drawdown_pct

    # ------------------------------------------------------------------
    # Position operations
    # ------------------------------------------------------------------

    def open_position(
        self, ticker: str, direction: str, price: float, size: float = 1.0
    ) -> dict:
        """Open a new position.

        Args:
            ticker: Ticker symbol.
            direction: "LONG" or "SHORT".
            price: Entry price.
            size: Position size as fraction of equity (default 1.0 = full position).

        Returns:
            Position dict.
        """
        if ticker in self._positions:
            self.logger.warning(f"Closing existing position for {ticker} before opening new one")
            self.close_position(ticker, price)

        position_value = self.cash * size
        quantity = position_value / price if price > 0 else 0

        position = {
            "ticker": ticker,
            "direction": direction,
            "entry_price": price,
            "entry_time": datetime.now(),
            "quantity": quantity,
            "position_value": position_value,
            "unrealized_pnl": 0.0,
            "exit_price": None,
            "exit_time": None,
            "realized_pnl": 0.0,
            "exit_reason": "",
            "model_version": "",
        }

        self._positions[ticker] = position
        self.logger.info(f"OPEN {direction} {ticker} @ ${price:.2f} (size: ${position_value:.2f})")

        return position

    def close_position(self, ticker: str, price: float, reason: str = "signal") -> Optional[dict]:
        """Close an existing position.

        Returns the completed trade dict, or None if no position exists.
        """
        if ticker not in self._positions:
            self.logger.warning(f"No position to close for {ticker}")
            return None

        position = self._positions.pop(ticker)
        direction = position["direction"]
        entry_price = position["entry_price"]
        quantity = position["quantity"]

        # Calculate P&L
        if direction == "LONG":
            pnl = (price - entry_price) * quantity
            pnl_pct = (price - entry_price) / entry_price
        else:  # SHORT
            pnl = (entry_price - price) * quantity
            pnl_pct = (entry_price - price) / entry_price

        # Update cash
        self.cash += pnl

        # Complete the trade record
        trade = {
            "ticker": ticker,
            "direction": direction,
            "entry_time": position["entry_time"],
            "entry_price": entry_price,
            "exit_time": datetime.now(),
            "exit_price": price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "exit_reason": reason,
            "loss_classification": "",
            "model_version": position.get("model_version", ""),
        }

        self._trade_history.append(trade)

        # Update peak equity
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        self._equity_curve.append((datetime.now(), self.equity))

        # Save to storage
        if self.storage is not None:
            try:
                trades_df = pd.DataFrame([trade])
                self.storage.save_trades(trades_df)
            except Exception as e:
                self.logger.error(f"Failed to save trade: {e}")

        self.logger.info(
            f"CLOSE {direction} {ticker} @ ${price:.2f} | "
            f"P&L: ${pnl:,.2f} ({pnl_pct*100:.2f}%) | "
            f"Reason: {reason}"
        )

        return trade

    def update_price(self, ticker: str, price: float) -> None:
        """Update the current price for mark-to-market P&L calculation."""
        if ticker not in self._positions:
            return

        position = self._positions[ticker]
        entry_price = position["entry_price"]
        quantity = position["quantity"]
        direction = position["direction"]

        if direction == "LONG":
            position["unrealized_pnl"] = (price - entry_price) * quantity
        else:
            position["unrealized_pnl"] = (entry_price - price) * quantity

        # Check drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_position(self, ticker: str) -> Optional[dict]:
        """Get the current position for a ticker."""
        return self._positions.get(ticker)

    def get_all_positions(self) -> Dict[str, dict]:
        """Get all open positions."""
        return dict(self._positions)

    def get_open_tickers(self) -> List[str]:
        """Get list of tickers with open positions."""
        return list(self._positions.keys())

    def is_flat(self) -> bool:
        """Check if there are no open positions."""
        return len(self._positions) == 0

    def get_trade_history(self, n: int = 50) -> List[dict]:
        """Get the last N trades."""
        return self._trade_history[-n:]

    def get_total_trades(self) -> int:
        """Get total number of closed trades."""
        return len(self._trade_history)

    def get_win_rate(self) -> float:
        """Calculate win rate from trade history."""
        if not self._trade_history:
            return 0.0
        wins = sum(1 for t in self._trade_history if t["pnl"] > 0)
        return wins / len(self._trade_history)

    def get_total_pnl(self) -> float:
        """Get total realized P&L."""
        return sum(t["pnl"] for t in self._trade_history)

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self._equity_curve:
            return pd.DataFrame(columns=["timestamp", "equity"])

        return pd.DataFrame(self._equity_curve, columns=["timestamp", "equity"])

    # ------------------------------------------------------------------
    # Risk management
    # ------------------------------------------------------------------

    def check_risk_limits(self) -> Tuple[bool, List[str]]:
        """Check all risk limits. Returns (all_ok, warnings)."""
        warnings = []

        if self.is_max_drawdown_exceeded:
            warnings.append(
                f"MAX DRAWDOWN EXCEEDED: {self.drawdown_pct*100:.1f}% "
                f"(limit: {self.max_drawdown_pct*100:.1f}%)"
            )

        # Check if any single position is too large
        for ticker, pos in self._positions.items():
            position_value = pos.get("position_value", 0)
            if self.equity > 0 and position_value / self.equity > 0.5:
                warnings.append(
                    f"Position size warning for {ticker}: "
                    f"{position_value/self.equity*100:.1f}% of equity"
                )

        return len(warnings) == 0, warnings

    def emergency_close_all(self, price_map: Dict[str, float]) -> List[dict]:
        """Close all positions immediately (e.g., on max drawdown)."""
        closed = []
        for ticker in list(self._positions.keys()):
            price = price_map.get(ticker, 0)
            if price > 0:
                trade = self.close_position(ticker, price, "emergency")
                if trade:
                    closed.append(trade)
        self.logger.warning(f"EMERGENCY: Closed all {len(closed)} positions")
        return closed

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Get comprehensive position and performance statistics."""
        return {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "drawdown_pct": round(self.drawdown_pct * 100, 2),
            "total_pnl": self.get_total_pnl(),
            "total_trades": self.get_total_trades(),
            "win_rate": round(self.get_win_rate() * 100, 2),
            "open_positions": len(self._positions),
            "open_tickers": self.get_open_tickers(),
        }
