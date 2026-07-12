"""
live/alerts.py — Signal, position, and risk alerts.

Outputs alerts to both console/log and can trigger
external notifications for critical events.
"""

from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional

from utils.config import get_config
from utils.logger import get_logger


class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    SIGNAL = "SIGNAL"


class Alerts:
    """Alert system for trading signals and risk events.

    Outputs to log + optional user-defined callbacks.
    """

    ALERT_COLORS = {
        AlertLevel.INFO: "\033[94m",     # Blue
        AlertLevel.WARNING: "\033[93m",  # Yellow
        AlertLevel.CRITICAL: "\033[91m", # Red
        AlertLevel.SIGNAL: "\033[92m",   # Green
    }
    RESET_COLOR = "\033[0m"

    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        self._callbacks: Dict[AlertLevel, List[Callable]] = {
            level: [] for level in AlertLevel
        }
        self._alert_history: List[dict] = []
        self._max_history = 1000

    def register_callback(self, level: AlertLevel, callback: Callable) -> None:
        """Register a callback function for a specific alert level.

        Callback signature: callback(level: AlertLevel, message: str, data: dict)
        """
        self._callbacks[level].append(callback)

    def send(self, level: AlertLevel, message: str, data: Optional[dict] = None) -> None:
        """Send an alert to log and all registered callbacks.

        Args:
            level: Alert severity level.
            message: Human-readable alert message.
            data: Optional structured data dict.
        """
        timestamp = datetime.now()
        color = self.ALERT_COLORS.get(level, "")

        # Log the alert
        log_message = f"[{level.value}] {message}"
        if level == AlertLevel.CRITICAL:
            self.logger.error(log_message)
        elif level == AlertLevel.WARNING:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

        # Colored console output
        print(f"{color}{timestamp.strftime('%H:%M:%S')} [{level.value}] {message}{self.RESET_COLOR}")

        # Store in history
        alert_record = {
            "timestamp": timestamp,
            "level": level.value,
            "message": message,
            "data": data or {},
        }
        self._alert_history.append(alert_record)

        # Trim history
        if len(self._alert_history) > self._max_history:
            self._alert_history = self._alert_history[-self._max_history:]

        # Fire callbacks
        for callback in self._callbacks[level]:
            try:
                callback(level, message, data)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")

    # ------------------------------------------------------------------
    # Pre-built alert methods
    # ------------------------------------------------------------------

    def signal_alert(self, ticker: str, action: str, price: float, confidence: float) -> None:
        """Alert on a new trading signal."""
        self.send(
            AlertLevel.SIGNAL,
            f"{ticker}: {action} signal @ ${price:.2f} (confidence: {confidence:.1%})",
            {
                "ticker": ticker,
                "action": action,
                "price": price,
                "confidence": confidence,
                "type": "signal",
            },
        )

    def position_opened(self, ticker: str, direction: str, price: float) -> None:
        """Alert on position opened."""
        self.send(
            AlertLevel.INFO,
            f"Position OPENED: {direction} {ticker} @ ${price:.2f}",
            {
                "ticker": ticker,
                "direction": direction,
                "price": price,
                "type": "position_open",
            },
        )

    def position_closed(
        self, ticker: str, direction: str, pnl: float, pnl_pct: float, reason: str
    ) -> None:
        """Alert on position closed."""
        level = AlertLevel.INFO if pnl >= 0 else AlertLevel.WARNING
        self.send(
            level,
            f"Position CLOSED: {direction} {ticker} | P&L: ${pnl:,.2f} ({pnl_pct*100:.2f}%) | {reason}",
            {
                "ticker": ticker,
                "direction": direction,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "reason": reason,
                "type": "position_close",
            },
        )

    def drawdown_warning(self, drawdown_pct: float, limit_pct: float) -> None:
        """Alert on approaching drawdown limit."""
        self.send(
            AlertLevel.WARNING,
            f"Drawdown warning: {drawdown_pct*100:.1f}% (limit: {limit_pct*100:.1f}%)",
            {
                "drawdown_pct": drawdown_pct,
                "limit_pct": limit_pct,
                "type": "drawdown",
            },
        )

    def max_drawdown_exceeded(self, drawdown_pct: float) -> None:
        """Critical alert when max drawdown is exceeded."""
        self.send(
            AlertLevel.CRITICAL,
            f"MAX DRAWDOWN EXCEEDED: {drawdown_pct*100:.1f}% — EMERGENCY CLOSE ALL",
            {
                "drawdown_pct": drawdown_pct,
                "type": "max_drawdown",
            },
        )

    def model_trained(self, ticker: str, version: str, metrics: dict) -> None:
        """Alert on model training completion."""
        self.send(
            AlertLevel.INFO,
            f"Model trained: {ticker} v{version} — Val loss: {metrics.get('val_loss', 'N/A')}",
            {
                "ticker": ticker,
                "version": version,
                "metrics": metrics,
                "type": "model_trained",
            },
        )

    def regime_change_detected(self, ticker: str, distance: float, regime_id: int) -> None:
        """Alert on regime change detection."""
        self.send(
            AlertLevel.WARNING,
            f"Regime change detected: {ticker} (distance={distance:.4f}, regime={regime_id})",
            {
                "ticker": ticker,
                "distance": distance,
                "regime_id": regime_id,
                "type": "regime_change",
            },
        )

    def forensics_complete(self, ticker: str, report: dict) -> None:
        """Alert on loss forensics analysis completion."""
        classification_counts = report.get("classification_counts", {})
        self.send(
            AlertLevel.INFO,
            f"Forensics complete: {ticker} — {report.get('total_trades_analyzed', 0)} trades analyzed",
            {
                "ticker": ticker,
                "classification_counts": classification_counts,
                "type": "forensics",
            },
        )

    def error_alert(self, component: str, error_message: str) -> None:
        """Alert on system errors."""
        self.send(
            AlertLevel.CRITICAL,
            f"ERROR in {component}: {error_message}",
            {
                "component": component,
                "error": error_message,
                "type": "error",
            },
        )

    # ------------------------------------------------------------------
    # History retrieval
    # ------------------------------------------------------------------

    def get_recent_alerts(self, n: int = 50, level: Optional[AlertLevel] = None) -> List[dict]:
        """Get recent alerts, optionally filtered by level."""
        alerts = self._alert_history
        if level is not None:
            alerts = [a for a in alerts if a["level"] == level.value]
        return alerts[-n:]

    def get_alerts_since(self, since: datetime) -> List[dict]:
        """Get alerts since a given timestamp."""
        return [a for a in self._alert_history if a["timestamp"] >= since]
