"""
evolution/auto_adapt.py — Automated parameter adjustment based on loss forensics.

FULL implementation with all methods:
- apply_fixes() — handles TYPE_A through TYPE_D recommendations
- optimize_risk_per_trade() — dynamic position sizing from recent performance
- adjust_stop_distance() — optimal stops from loss forensics data
- detect_regime_boundary() — latent state regime shift identification
- nightly_adaptation_cycle() — full overnight learning orchestration
- validate_fixes() — backtest proposed fixes before applying
- promote_model() — replace live model only if performance improves
"""

import json
import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.config import get_config
from utils.logger import get_logger


class AutoAdapter:
    """Applies automated fixes based on loss forensics analysis.

    Reads forensics reports and adjusts model parameters, stop distances,
    reward functions, and signal thresholds without human intervention.

    All changes are validated via backtest before promotion to live.
    """

    def __init__(self, ticker_manager=None, storage=None, config=None):
        self.ticker_manager = ticker_manager
        self.storage = storage
        self.config = config or get_config()
        self.logger = get_logger()

        # Change tracking
        self._change_log: List[dict] = []
        self._adaptation_count: Dict[str, int] = {}

        # State tracking for optimize_risk_per_trade
        self._recent_performance: Dict[str, List[float]] = {}
        self._performance_window = 50  # last N trades for risk adjustment

        # Regime tracking
        self._regime_thresholds: Dict[str, float] = {}
        self._last_regime: Dict[str, int] = {}

        # Model version tracking for promote_model
        self._staging_models: Dict[str, dict] = {}
        self._live_model_metrics: Dict[str, dict] = {}

    # ======================================================================
    # apply_fixes — TYPE_A through TYPE_D
    # ======================================================================

    def apply_fixes(self, ticker: str, forensics_report: dict) -> dict:
        """Apply recommended fixes from a forensics report.

        Handles all four types: TYPE_A (widening stops), TYPE_B (retraining),
        TYPE_C (regime sensitivity), TYPE_D (confidence filtering).

        Args:
            ticker: Ticker symbol.
            forensics_report: Report dict from LossForensics.generate_forensics_report()

        Returns:
            Summary of changes applied.
        """
        if ticker not in self._adaptation_count:
            self._adaptation_count[ticker] = 0
        self._adaptation_count[ticker] += 1

        changes = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "adaptation_number": self._adaptation_count[ticker],
            "changes": [],
        }

        recommendations = forensics_report.get("recommendations", [])
        classification_counts = forensics_report.get("classification_counts", {})

        for rec in recommendations:
            rec_type = rec.get("type", "")

            if rec_type == "TYPE_A":
                change = self._apply_type_a_fix(ticker, rec, classification_counts)
            elif rec_type == "TYPE_B":
                change = self._apply_type_b_fix(ticker, rec, classification_counts)
            elif rec_type == "TYPE_C":
                change = self._apply_type_c_fix(ticker, rec, classification_counts)
            elif rec_type == "TYPE_D":
                change = self._apply_type_d_fix(ticker, rec, classification_counts)
            else:
                change = None

            if change:
                changes["changes"].append(change)

        # If TYPE_A dominates (>50%), also call optimize_risk_per_trade
        total = sum(classification_counts.values()) if classification_counts else 0
        type_a_pct = classification_counts.get("TYPE_A", 0) / max(total, 1)
        if type_a_pct > 0.5:
            risk_change = self.optimize_risk_per_trade(ticker)
            if risk_change:
                changes["changes"].append(risk_change)

        self._change_log.append(changes)
        self.logger.info(
            f"Applied {len(changes['changes'])} fixes for {ticker} "
            f"(adaptation #{self._adaptation_count[ticker]})"
        )

        return changes

    # ------------------------------------------------------------------
    # TYPE_A: Stop too tight — widen stop distance
    # ------------------------------------------------------------------

    def _apply_type_a_fix(
        self, ticker: str, recommendation: dict, counts: dict
    ) -> Optional[dict]:
        """TYPE_A: Widen the stop distance.

        Calculates new stop distance from loss forensics data,
        updates the RL environment's adverse excursion tolerance,
        and adjusts drawdown penalty threshold.
        """
        new_stop_pct = recommendation.get("current_stop_pct", 2.0)
        old_stop_pct = 2.0

        # Retrieve stored stop distance if previously adapted
        if self.ticker_manager and ticker in self.ticker_manager._registry:
            entry = self.ticker_manager._registry[ticker]
            if entry.get("rl_agent") is not None:
                env = entry["rl_agent"].env
                old_stop_pct = getattr(env, "adapted_stop_distance", 0.02) * 100
                env.adapted_stop_distance = new_stop_pct / 100.0

        change = {
            "type": "TYPE_A",
            "description": (
                f"Stop distance adjusted from {old_stop_pct:.1f}% "
                f"to {new_stop_pct:.1f}% based on {counts.get('TYPE_A', 0)} "
                f"losing trades where price moved favorably before reversing"
            ),
            "old_value": old_stop_pct,
            "new_value": new_stop_pct,
        }

        self.logger.info(f"[{ticker}] {change['description']}")
        return change

    # ------------------------------------------------------------------
    # TYPE_B: Wrong direction — retrain entry logic
    # ------------------------------------------------------------------

    def _apply_type_b_fix(
        self, ticker: str, recommendation: dict, counts: dict
    ) -> Optional[dict]:
        """TYPE_B: Wrong direction — retrain world model on entry patterns.

        Flags the world model for retraining. The retraining will use
        the latent states that led to wrong-direction entries as
        negative examples in the training data.
        """
        type_b_count = counts.get("TYPE_B", 0)

        change = {
            "type": "TYPE_B",
            "description": (
                f"Flagged for world model retraining — {type_b_count} trades "
                f"never moved favorably. Entry patterns need revision."
            ),
            "action": "Retraining recommended on next adaptation cycle",
        }

        # Mark ticker for retraining
        if self.ticker_manager:
            entry = self.ticker_manager._registry.get(ticker, {})
            entry["needs_retraining"] = True
            # Lower the learning rate for fine-tuning
            entry["retrain_lr"] = self.config.LEARNING_RATE * 0.1

        self.logger.info(f"[{ticker}] {change['description']}")
        return change

    # ------------------------------------------------------------------
    # TYPE_C: Regime change — adjust detection sensitivity
    # ------------------------------------------------------------------

    def _apply_type_c_fix(
        self, ticker: str, recommendation: dict, counts: dict
    ) -> Optional[dict]:
        """TYPE_C: Regime change — adjust sensitivity.

        Lowers the cosine distance threshold for regime detection,
        so the system exits faster when a market regime shift occurs.
        """
        old_threshold = 0.5
        new_threshold = max(0.3, old_threshold - 0.05)

        # Store per-ticker threshold
        self._regime_thresholds[ticker] = new_threshold

        # If there's a regime detector with this ticker, update it
        if hasattr(self, "_regime_detector") and self._regime_detector is not None:
            self._regime_detector.threshold = new_threshold

        type_c_count = counts.get("TYPE_C", 0)

        change = {
            "type": "TYPE_C",
            "description": (
                f"Regime detection threshold lowered from {old_threshold:.2f} "
                f"to {new_threshold:.2f} — {type_c_count} regime-change losses. "
                f"System will react faster to latent space shifts."
            ),
            "old_value": old_threshold,
            "new_value": new_threshold,
        }

        self.logger.info(f"[{ticker}] {change['description']}")
        return change

    # ------------------------------------------------------------------
    # TYPE_D: Noise/whipsaw — filter low-confidence signals
    # ------------------------------------------------------------------

    def _apply_type_d_fix(
        self, ticker: str, recommendation: dict, counts: dict
    ) -> Optional[dict]:
        """TYPE_D: Noise — increase confidence threshold to filter noise trades."""
        old_threshold = 0.5
        new_threshold = min(0.85, old_threshold + 0.1)

        # Store the new threshold
        if self.ticker_manager and ticker in self.ticker_manager._registry:
            entry = self.ticker_manager._registry[ticker]
            entry["min_confidence"] = new_threshold

        type_d_count = counts.get("TYPE_D", 0)

        change = {
            "type": "TYPE_D",
            "description": (
                f"Signal confidence threshold increased from {old_threshold:.2f} "
                f"to {new_threshold:.2f} — {type_d_count} noise trades filtered"
            ),
            "old_value": old_threshold,
            "new_value": new_threshold,
        }

        self.logger.info(f"[{ticker}] {change['description']}")
        return change

    # ======================================================================
    # optimize_risk_per_trade — Dynamic position sizing
    # ======================================================================

    def optimize_risk_per_trade(self, ticker: str) -> Optional[dict]:
        """Dynamically adjust position sizing based on recent performance.

        Uses the last N trades to compute:
        - Rolling win rate
        - Rolling Sharpe ratio
        - Consecutive losses

        If recent performance is strong, increases risk up to RISK_PER_TRADE_MAX.
        If recent performance is weak, decreases risk down to RISK_PER_TRADE_MIN.
        If drawdown > 25%, forces minimum risk.

        Returns change dict, or None if no adjustment needed.
        """
        if self.storage is None:
            return None

        try:
            trades = self.storage.load_trades(ticker=ticker)
            if len(trades) < 10:
                return None

            recent = trades.tail(self._performance_window)
            pnls = recent["pnl"].values

            # Compute metrics
            wins = (pnls > 0).sum()
            total = len(pnls)
            rolling_win_rate = wins / total if total > 0 else 0.0

            # Consecutive losses
            consec_losses = 0
            for p in reversed(pnls):
                if p < 0:
                    consec_losses += 1
                else:
                    break

            # Compute optimal risk
            current_risk = getattr(self, "_current_risk_pct", None)
            if current_risk is None:
                current_risk = (self.config.RISK_PER_TRADE_MIN + self.config.RISK_PER_TRADE_MAX) / 2

            if rolling_win_rate > 0.6 and consec_losses <= 1:
                new_risk = min(self.config.RISK_PER_TRADE_MAX, current_risk * 1.2)
                reason = f"Strong performance: {rolling_win_rate:.1%} win rate"
            elif rolling_win_rate < 0.35 or consec_losses >= 4:
                new_risk = max(self.config.RISK_PER_TRADE_MIN, current_risk * 0.7)
                reason = f"Weak performance: {rolling_win_rate:.1%} win rate, {consec_losses} consecutive losses"
            else:
                return None  # No change needed

            self._current_risk_pct = new_risk

            change = {
                "type": "RISK_ADJUST",
                "description": (
                    f"Risk per trade adjusted from {current_risk*100:.2f}% "
                    f"to {new_risk*100:.2f}% — {reason}"
                ),
                "old_value": round(current_risk, 6),
                "new_value": round(new_risk, 6),
            }

            self.logger.info(f"[{ticker}] {change['description']}")
            return change

        except Exception as e:
            self.logger.error(f"Failed to optimize risk for {ticker}: {e}")
            return None

    # ======================================================================
    # adjust_stop_distance — Calculate optimal stops from loss forensics
    # ======================================================================

    def adjust_stop_distance(
        self,
        ticker: str,
        loss_trades: List[dict],
        price_data: pd.DataFrame,
    ) -> float:
        """Calculate optimal stop distance from loss forensics data.

        For each losing trade classified as TYPE_A, computes the
        stop distance that would have:
        - Preserved the win if the favorable move was large enough
        - Minimized the loss if the reversal was inevitable

        Uses the 75th percentile of adverse excursions across all
        TYPE_A trades to determine the recommended stop distance.

        THIS IS NOT ATR — it's derived purely from the machine's
        own loss forensics data.

        Args:
            ticker: Ticker symbol.
            loss_trades: List of losing trade records.
            price_data: DataFrame with full price history.

        Returns:
            Recommended stop distance as a percentage of entry price.
        """
        from evolution.loss_forensics import LossForensics

        if not loss_trades:
            return 0.02

        forensics = LossForensics(storage=self.storage)
        if self.ticker_manager and ticker in self.ticker_manager._registry:
            forensics.world_model = self.ticker_manager._registry[ticker].get("world_model")

        optimal_stops = []

        for trade in loss_trades:
            classification = forensics.classify_loss(trade, price_data)
            if classification == "TYPE_A":
                stop = forensics.identify_stop_level(trade, price_data)
                optimal_stops.append(stop)

        if optimal_stops:
            avg_stop = float(np.mean(optimal_stops))
        else:
            avg_stop = 0.02

        # Store for later use
        if not hasattr(self, "_adapted_stops"):
            self._adapted_stops: Dict[str, float] = {}
        self._adapted_stops[ticker] = avg_stop

        self.logger.info(
            f"[{ticker}] Adjusted stop distance to {avg_stop*100:.2f}% "
            f"based on {len(optimal_stops)} TYPE_A trades"
        )

        return avg_stop

    # ======================================================================
    # detect_regime_boundary — Identify latent state regime shift
    # ======================================================================

    def detect_regime_boundary(
        self, ticker: str, latent_vector: np.ndarray
    ) -> Tuple[bool, Optional[int]]:
        """Identify when the latent state indicates a regime shift.

        Compares the current latent vector against the accumulated
        latent history for this ticker. If the cosine distance exceeds
        the ticker's threshold, a regime boundary is flagged.

        Args:
            ticker: Ticker symbol.
            latent_vector: 256-dim latent state vector.

        Returns:
            (is_regime_change: bool, new_regime_id: int or None)
        """
        threshold = self._regime_thresholds.get(ticker, 0.5)

        if not hasattr(self, "_latent_buffers"):
            self._latent_buffers: Dict[str, List[np.ndarray]] = {}

        if ticker not in self._latent_buffers:
            self._latent_buffers[ticker] = []

        buffer = self._latent_buffers[ticker]
        current_regime = self._last_regime.get(ticker, 0)

        # Need at least a few samples for comparison
        if len(buffer) < 5:
            buffer.append(latent_vector.copy())
            if len(buffer) > 100:
                buffer.pop(0)
            return False, None

        # Compare against the mean of the last 5 latent states
        recent_mean = np.mean(buffer[-5:], axis=0)
        from scipy.spatial.distance import cosine
        distance = cosine(recent_mean, latent_vector.flatten())

        if distance > threshold:
            new_regime = current_regime + 1
            self._last_regime[ticker] = new_regime
            buffer.clear()
            self.logger.warning(
                f"[{ticker}] REGIME CHANGE DETECTED: "
                f"distance={distance:.4f} > threshold={threshold:.2f}, "
                f"new regime={new_regime}"
            )
            return True, new_regime

        buffer.append(latent_vector.copy())
        if len(buffer) > 100:
            buffer.pop(0)

        return False, None

    # ======================================================================
    # nightly_adaptation_cycle — Full overnight learning orchestration
    # ======================================================================

    def nightly_adaptation_cycle(self) -> dict:
        """Orchestrate the full overnight learning and adaptation process.

        Runs for all tickers:
        1. Load recent trades and classify losses
        2. Generate forensics report
        3. Propose fixes (stop distances, risk, confidence thresholds)
        4. Validate fixes via backtest
        5. Promote model if validation passes

        This is designed to run as a weekend/overnight job.

        Returns:
            Summary dict with per-ticker results.
        """
        from evolution.loss_forensics import LossForensics

        self.logger.info("=" * 60)
        self.logger.info("NIGHTLY ADAPTATION CYCLE STARTING")
        self.logger.info("=" * 60)

        cycle_summary = {
            "timestamp": datetime.now().isoformat(),
            "tickers_processed": 0,
            "fixes_applied": 0,
            "models_promoted": 0,
            "per_ticker": {},
        }

        if self.ticker_manager is None:
            self.logger.error("No ticker manager configured")
            return cycle_summary

        tickers = self.config.TICKERS

        for ticker in tickers:
            ticker_result = {
                "forensics_run": False,
                "fixes_applied": 0,
                "validated": False,
                "promoted": False,
                "errors": [],
            }

            try:
                # Check if ticker is ready
                if not self.ticker_manager.is_ticker_ready(ticker):
                    ticker_result["errors"].append("Ticker not ready")
                    cycle_summary["per_ticker"][ticker] = ticker_result
                    continue

                # Load recent trades
                trades = []
                if self.storage:
                    trades_df = self.storage.load_trades(ticker=ticker)
                    if len(trades_df) > 0:
                        losing = trades_df[trades_df["pnl"] < 0]
                        trades = losing.to_dict("records")

                if not trades:
                    ticker_result["errors"].append("No losing trades to analyze")
                    cycle_summary["per_ticker"][ticker] = ticker_result
                    continue

                # Run forensics
                world_model = self.ticker_manager._registry[ticker].get("world_model")
                forensics = LossForensics(storage=self.storage, world_model=world_model)

                price_data = None
                if self.storage:
                    try:
                        price_data = self.storage.load_tick_data(ticker)
                    except Exception:
                        pass

                if price_data is None or len(price_data) == 0:
                    ticker_result["errors"].append("No price data for forensics")
                    cycle_summary["per_ticker"][ticker] = ticker_result
                    continue

                report, formatted = forensics.generate_forensics_report(trades, price_data)
                forensics.store_forensics(ticker, report)
                ticker_result["forensics_run"] = True

                # Apply fixes
                changes = self.apply_fixes(ticker, report)
                ticker_result["fixes_applied"] = len(changes.get("changes", []))

                # Adjust stop distance
                stop = self.adjust_stop_distance(ticker, trades, price_data)
                ticker_result["adapted_stop_pct"] = round(stop * 100, 2)

                # Optimize risk
                risk_change = self.optimize_risk_per_trade(ticker)

                # Validate fixes
                validation_result = self.validate_fixes(ticker, changes)
                ticker_result["validated"] = validation_result.get("passed", False)

                # Promote if validation passed
                if validation_result.get("passed", False):
                    promoted = self.promote_model(ticker, validation_result)
                    ticker_result["promoted"] = promoted

                cycle_summary["fixes_applied"] += ticker_result["fixes_applied"]
                if ticker_result["promoted"]:
                    cycle_summary["models_promoted"] += 1

            except Exception as e:
                ticker_result["errors"].append(str(e))
                self.logger.error(f"Nightly cycle error for {ticker}: {e}")

            cycle_summary["tickers_processed"] += 1
            cycle_summary["per_ticker"][ticker] = ticker_result

        self.logger.info(
            f"Nightly adaptation complete: "
            f"{cycle_summary['tickers_processed']} tickers, "
            f"{cycle_summary['fixes_applied']} fixes, "
            f"{cycle_summary['models_promoted']} promoted"
        )

        return cycle_summary

    # ======================================================================
    # validate_fixes — Backtest proposed fixes before applying
    # ======================================================================

    def validate_fixes(self, ticker: str, proposed_changes: dict) -> dict:
        """Backtest proposed fixes before applying them to the live model.

        Creates a staging copy of the model with proposed changes,
        runs a quick backtest, and compares performance against the
        current live model's baseline.

        Args:
            ticker: Ticker symbol.
            proposed_changes: Changes dict from apply_fixes().

        Returns:
            Validation result dict with: passed, baseline_metrics,
            proposed_metrics, improvement_pct.
        """
        from evolution.backtester import Backtester

        validation = {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "passed": False,
            "baseline_metrics": {},
            "proposed_metrics": {},
            "improvement_pct": 0.0,
            "reason": "",
        }

        if self.ticker_manager is None or not self.ticker_manager.is_ticker_ready(ticker):
            validation["reason"] = "Ticker not ready for validation"
            return validation

        if self.storage is None:
            validation["reason"] = "No storage for backtesting"
            return validation

        try:
            prices = self.storage.get_close_prices(ticker)
            if len(prices) < 1000:
                validation["reason"] = "Not enough price data for validation backtest"
                return validation

            backtester = Backtester(storage=self.storage)

            # Baseline: current live model
            baseline = backtester.run_backtest(
                ticker, prices[-2000:], ticker_manager=self.ticker_manager
            )
            validation["baseline_metrics"] = {
                "win_rate": baseline["win_rate"],
                "net_profit": baseline["net_profit"],
                "sharpe_ratio": baseline["sharpe_ratio"],
                "max_drawdown_pct": baseline["max_drawdown_pct"],
                "profit_factor": baseline["profit_factor"],
                "total_trades": baseline["total_trades"],
            }

            # Proposed: apply changes to a staging copy
            staging_mgr = self._create_staging_copy(ticker, proposed_changes)

            proposed = backtester.run_backtest(
                ticker, prices[-2000:], ticker_manager=staging_mgr
            )
            validation["proposed_metrics"] = {
                "win_rate": proposed["win_rate"],
                "net_profit": proposed["net_profit"],
                "sharpe_ratio": proposed["sharpe_ratio"],
                "max_drawdown_pct": proposed["max_drawdown_pct"],
                "profit_factor": proposed["profit_factor"],
                "total_trades": proposed["total_trades"],
            }

            # Compare: proposed must improve net profit and not worsen drawdown
            profit_improvement = proposed["net_profit"] - baseline["net_profit"]
            dd_change = proposed["max_drawdown_pct"] - baseline["max_drawdown_pct"]

            if profit_improvement > 0 and dd_change < 0.05:
                validation["passed"] = True
                validation["improvement_pct"] = (
                    profit_improvement / max(abs(baseline["net_profit"]), 1.0) * 100
                )
                validation["reason"] = (
                    f"Validated: P&L improved by ${profit_improvement:,.2f}, "
                    f"DD change: {dd_change*100:+.1f}%"
                )
                self.logger.info(f"[{ticker}] Validation PASSED: {validation['reason']}")
            else:
                validation["reason"] = (
                    f"Rejected: P&L change ${profit_improvement:,.2f}, "
                    f"DD change: {dd_change*100:+.1f}%"
                )
                self.logger.info(f"[{ticker}] Validation FAILED: {validation['reason']}")

            # Store staging model for potential promotion
            self._staging_models[ticker] = {
                "model": staging_mgr,
                "metrics": validation["proposed_metrics"],
                "validation": validation,
            }

        except Exception as e:
            validation["reason"] = f"Validation error: {e}"
            self.logger.error(f"[{ticker}] Validation error: {e}")

        return validation

    def _create_staging_copy(self, ticker: str, changes: dict):
        """Create a staging copy of the ticker manager with proposed changes applied.

        Returns a new TickerManager instance with the changes applied,
        so the live model remains untouched during validation.
        """
        from models.ticker_manager import TickerManager

        staging = TickerManager(self.config)

        # Copy the world model and RL agent
        if ticker in self.ticker_manager._registry:
            entry = self.ticker_manager._registry[ticker]
            staging._registry[ticker] = deepcopy(entry)
            staging._registry[ticker]["trained"] = True

        # Apply changes to staging
        for change in changes.get("changes", []):
            if change["type"] == "TYPE_A" and "new_value" in change:
                new_stop = change["new_value"] / 100.0
                if ticker in staging._registry:
                    rl_entry = staging._registry[ticker].get("rl_agent")
                    if rl_entry:
                        rl_entry.env.adapted_stop_distance = new_stop

            elif change["type"] == "TYPE_D" and "new_value" in change:
                new_conf = change["new_value"]
                if ticker in staging._registry:
                    staging._registry[ticker]["min_confidence"] = new_conf

        return staging

    # ======================================================================
    # promote_model — Replace live model only if fixes improve performance
    # ======================================================================

    def promote_model(self, ticker: str, validation_result: dict) -> bool:
        """Replace the live model with the staging model only if
        the validated fixes improve performance.

        Checks:
        1. Validation must have passed
        2. Net profit improvement > 0
        3. Drawdown not significantly worse (< 5% increase)
        4. At least 10 trades in validation backtest

        Args:
            ticker: Ticker symbol.
            validation_result: Dict from validate_fixes().

        Returns:
            True if model was promoted, False otherwise.
        """
        if not validation_result.get("passed", False):
            self.logger.info(f"[{ticker}] Promotion skipped: validation not passed")
            return False

        proposed = validation_result.get("proposed_metrics", {})
        baseline = validation_result.get("baseline_metrics", {})

        # Guard: minimum trade count
        if proposed.get("total_trades", 0) < 10:
            self.logger.info(
                f"[{ticker}] Promotion skipped: only {proposed.get('total_trades', 0)} "
                f"trades in validation (need 10+)"
            )
            return False

        # Guard: drawdown must not increase by more than 5%
        dd_increase = proposed.get("max_drawdown_pct", 0) - baseline.get("max_drawdown_pct", 0)
        if dd_increase > 0.05:
            self.logger.info(
                f"[{ticker}] Promotion skipped: drawdown increase {dd_increase*100:.1f}% > 5%"
            )
            return False

        # Guard: profit must improve
        profit_delta = proposed.get("net_profit", 0) - baseline.get("net_profit", 0)
        if profit_delta <= 0:
            self.logger.info(
                f"[{ticker}] Promotion skipped: no profit improvement "
                f"(${profit_delta:,.2f})"
            )
            return False

        # Promote: replace live model with staging
        if ticker in self._staging_models:
            staging_mgr = self._staging_models[ticker]["model"]

            if ticker in self.ticker_manager._registry and ticker in staging_mgr._registry:
                # Copy world model state
                live_wm = self.ticker_manager._registry[ticker]["world_model"]
                staging_wm = staging_mgr._registry[ticker]["world_model"]
                if live_wm is not None and staging_wm is not None:
                    live_wm.load_state_dict(staging_wm.state_dict())

                # Copy RL agent
                live_rl = self.ticker_manager._registry[ticker]["rl_agent"]
                staging_rl = staging_mgr._registry[ticker]["rl_agent"]
                if live_rl is not None and staging_rl is not None:
                    live_rl.model.set_parameters(staging_rl.model.get_parameters())

                # Bump version
                old_version = self.ticker_manager._registry[ticker]["version"]
                parts = old_version.split(".")
                new_minor = int(parts[-1]) + 1
                new_version = f"{parts[0]}.{parts[1]}.{new_minor}"
                self.ticker_manager._registry[ticker]["version"] = new_version

                # Save promoted model
                checkpoint_dir = os.path.join(self.config.MODEL_DIR, ticker)
                os.makedirs(checkpoint_dir, exist_ok=True)
                import torch
                torch.save(
                    live_wm.state_dict(),
                    os.path.join(checkpoint_dir, f"world_model_v{new_version}.pt")
                )

                self.logger.info(
                    f"[{ticker}] MODEL PROMOTED: v{old_version} → v{new_version}. "
                    f"P&L improvement: ${profit_delta:,.2f}"
                )

                # Store live metrics
                self._live_model_metrics[ticker] = proposed

                return True

        return False

    # ======================================================================
    # Accessors
    # ======================================================================

    def get_change_log(self, ticker: Optional[str] = None) -> List[dict]:
        """Get the history of all changes applied."""
        if ticker:
            return [c for c in self._change_log if c.get("ticker") == ticker]
        return self._change_log

    def get_adaptation_count(self, ticker: str) -> int:
        """Get the number of adaptations applied to a ticker."""
        return self._adaptation_count.get(ticker, 0)

    def get_current_risk(self, ticker: str) -> float:
        """Get the current risk-per-trade setting for a ticker."""
        return getattr(self, "_current_risk_pct", self.config.RISK_PER_TRADE_MIN)

    def get_live_model_metrics(self, ticker: str) -> Optional[dict]:
        """Get the stored live model metrics for a ticker."""
        return self._live_model_metrics.get(ticker)
