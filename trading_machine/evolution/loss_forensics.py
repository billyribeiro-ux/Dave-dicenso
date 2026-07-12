"""
evolution/loss_forensics.py — Classifies every losing trade into 4 types.

TYPE_A — STOP TOO TIGHT: price moved favorably then reversed
TYPE_B — WRONG DIRECTION: price never moved favorably
TYPE_C — REGIME CHANGE: trade was valid but market shifted
TYPE_D — NOISE/WHIPSAW: pattern was random

The machine discovers its own optimal stop distance — NOT ATR.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cosine

from utils.logger import get_logger


class LossForensics:
    """Analyzes losing trades to classify failure modes and recommend fixes.

    Operates on raw price data and world model latent states — no human concepts.
    """

    COSINE_DISTANCE_THRESHOLD = 0.5  # Latent space shift threshold for regime change
    FAVORABLE_MOVE_MIN_PCT = 0.001  # 0.1% minimum favorable move to count
    NOISE_STD_MULTIPLIER = 1.0  # Within 1 std dev = noise

    def __init__(self, storage=None, world_model=None):
        self.storage = storage
        self.world_model = world_model
        self.logger = get_logger()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    # ------------------------------------------------------------------
    # Individual trade classification
    # ------------------------------------------------------------------

    def classify_loss(
        self, trade_record: dict, full_price_history: pd.DataFrame
    ) -> str:
        """Classify a losing trade into TYPE_A, TYPE_B, TYPE_C, or TYPE_D.

        Args:
            trade_record: dict with ticker, direction, entry_time, entry_price,
                         exit_time, exit_price, pnl
            full_price_history: DataFrame with all price data including post-trade period.

        Returns:
            Classification string: "TYPE_A", "TYPE_B", "TYPE_C", or "TYPE_D"
        """
        entry_time = trade_record.get("entry_time")
        exit_time = trade_record.get("exit_time")
        entry_price = trade_record.get("entry_price", 0)
        exit_price = trade_record.get("exit_price", 0)
        direction = trade_record.get("direction", "")
        pnl = trade_record.get("pnl", 0)

        if entry_price == 0:
            return "TYPE_D"

        # Extract price data during the trade
        if full_price_history is not None and "timestamp" in full_price_history.columns:
            if isinstance(entry_time, str):
                entry_time = pd.Timestamp(entry_time)
            if isinstance(exit_time, str):
                exit_time = pd.Timestamp(exit_time)

            trade_prices = full_price_history[
                (full_price_history["timestamp"] >= entry_time) &
                (full_price_history["timestamp"] <= exit_time)
            ]["close"].values
        else:
            # If no price history, can only do basic classification
            trade_prices = np.array([])

        # Also get post-trade prices for regime change detection
        post_trade_prices = np.array([])
        if full_price_history is not None and "timestamp" in full_price_history.columns:
            post_data = full_price_history[
                full_price_history["timestamp"] > exit_time
            ]["close"].values
            if len(post_data) > 0:
                post_trade_prices = post_data[:min(200, len(post_data))]

        # --- TYPE_B check first (wrong direction) ---
        is_wrong_direction = self._check_wrong_direction(
            trade_prices, entry_price, direction
        )
        if is_wrong_direction:
            # Encode the entry window for storage
            if self.world_model is not None and len(trade_prices) > 0:
                self._encode_entry_latent(trade_prices[:500] if len(trade_prices) >= 500 else trade_prices)
            return "TYPE_B"

        # --- TYPE_A check (stop too tight) ---
        is_stop_too_tight, missed_profit = self._check_stop_too_tight(
            trade_prices, entry_price, exit_price, direction
        )
        if is_stop_too_tight:
            trade_record["missed_profit"] = missed_profit
            return "TYPE_A"

        # --- TYPE_C check (regime change) ---
        is_regime_change, regime_shift_ts = self._check_regime_change(
            trade_prices, post_trade_prices, entry_price, direction
        )
        if is_regime_change:
            trade_record["regime_shift_timestamp"] = str(regime_shift_ts) if regime_shift_ts else None
            return "TYPE_C"

        # --- TYPE_D (noise/whipsaw) ---
        return "TYPE_D"

    def _check_wrong_direction(
        self, prices: np.ndarray, entry_price: float, direction: str
    ) -> bool:
        """Check if price EVER moved favorably. If never, it's TYPE_B."""
        if len(prices) == 0:
            return True

        threshold = entry_price * self.FAVORABLE_MOVE_MIN_PCT

        if direction == "LONG":
            favorable = prices > entry_price + threshold
        elif direction == "SHORT":
            favorable = prices < entry_price - threshold
        else:
            return True

        return not favorable.any()

    def _check_stop_too_tight(
        self,
        prices: np.ndarray,
        entry_price: float,
        exit_price: float,
        direction: str,
    ) -> Tuple[bool, float]:
        """Check if price moved favorably by 2x the adverse excursion before reversing.

        Returns (is_stop_too_tight, missed_profit_amount).
        """
        if len(prices) < 2:
            return False, 0.0

        if direction == "LONG":
            favorable_extreme = prices.max()
            adverse_extreme = prices.min()
            favorable_move = (favorable_extreme - entry_price) / entry_price
            adverse_move = (entry_price - adverse_extreme) / entry_price
            missed_profit = (favorable_extreme - exit_price) / entry_price * 100.0
        elif direction == "SHORT":
            favorable_extreme = prices.min()
            adverse_extreme = prices.max()
            favorable_move = (entry_price - favorable_extreme) / entry_price
            adverse_move = (adverse_extreme - entry_price) / entry_price
            missed_profit = (exit_price - favorable_extreme) / entry_price * 100.0
        else:
            return False, 0.0

        # Price moved favorably by at least 2x the adverse excursion
        if adverse_move > 0 and favorable_move >= 2.0 * adverse_move:
            return True, missed_profit

        return False, 0.0

    def _check_regime_change(
        self,
        trade_prices: np.ndarray,
        post_prices: np.ndarray,
        entry_price: float,
        direction: str,
    ) -> Tuple[bool, Optional[int]]:
        """Check if there was a significant latent space shift (regime change).

        Encodes price windows before and after the suspected shift,
        computes cosine distance between latent vectors.
        If distance > 0.5, it's TYPE_C.
        """
        if self.world_model is None:
            return False, None

        if len(trade_prices) < 500:
            return False, None

        # Split trade window into first half and second half
        mid = len(trade_prices) // 2
        first_half = trade_prices[:500] if len(trade_prices) >= 500 else trade_prices[:mid]
        second_half = trade_prices[-500:] if len(trade_prices) >= 500 else trade_prices[mid:]

        if len(first_half) < 100 or len(second_half) < 100:
            return False, None

        try:
            # Encode both halves
            latent1 = self._encode_window(first_half)
            latent2 = self._encode_window(second_half)

            if latent1 is None or latent2 is None:
                return False, None

            # Cosine distance
            distance = cosine(latent1, latent2)

            if distance > self.COSINE_DISTANCE_THRESHOLD:
                return True, mid
        except Exception:
            pass

        return False, None

    def _encode_window(self, prices: np.ndarray) -> Optional[np.ndarray]:
        """Encode a price window to latent vector."""
        if self.world_model is None:
            return None

        try:
            prices = prices[-500:].astype(np.float32)
            if len(prices) < 500:
                prices = np.pad(prices, (500 - len(prices), 0), mode="edge")

            mean = prices.mean()
            std = prices.std()
            if std < 1e-8:
                std = 1.0
            normalized = (prices - mean) / std

            x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            x = x.to(self.device)

            self.world_model.eval()
            self.world_model.to(self.device)

            with torch.no_grad():
                mu, logvar, z = self.world_model.encode(x)

            return z.cpu().numpy().flatten()
        except Exception as e:
            self.logger.warning(f"Failed to encode window: {e}")
            return None

    def _encode_entry_latent(self, prices: np.ndarray) -> Optional[np.ndarray]:
        """Encode the price window at entry time for TYPE_B storage."""
        return self._encode_window(prices)

    # ------------------------------------------------------------------
    # Batch analysis
    # ------------------------------------------------------------------

    def analyze_batch(
        self, trades_list: List[dict], price_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Classify all losing trades in a batch.

        Returns DataFrame with columns: trade_id, classification, details_json, recommended_action.
        """
        results = []
        for i, trade in enumerate(trades_list):
            classification = self.classify_loss(trade, price_data)
            recommended = self._get_recommendation(classification)

            details = {
                "pnl": trade.get("pnl", 0),
                "direction": trade.get("direction", ""),
                "entry_price": trade.get("entry_price", 0),
                "exit_price": trade.get("exit_price", 0),
                "missed_profit": trade.get("missed_profit"),
                "regime_shift_timestamp": trade.get("regime_shift_timestamp"),
            }

            results.append({
                "trade_id": i,
                "classification": classification,
                "details_json": json.dumps(details),
                "recommended_action": recommended,
            })

        df = pd.DataFrame(results)
        if len(df) > 0:
            self.logger.info(
                f"Batch analysis: {len(df)} trades — "
                f"A: {(df['classification']=='TYPE_A').sum()}, "
                f"B: {(df['classification']=='TYPE_B').sum()}, "
                f"C: {(df['classification']=='TYPE_C').sum()}, "
                f"D: {(df['classification']=='TYPE_D').sum()}"
            )
        return df

    def _get_recommendation(self, classification: str) -> str:
        """Get recommended action for each classification type."""
        if classification == "TYPE_A":
            return "Widen stop distance based on favorable excursion distribution"
        elif classification == "TYPE_B":
            return "Retrain world model on entry patterns; review latent state"
        elif classification == "TYPE_C":
            return "Add regime detection filter; increase regime change sensitivity"
        elif classification == "TYPE_D":
            return "Reduce position size in high-noise environments; ignore pattern"
        return "Unknown"

    # ------------------------------------------------------------------
    # Optimal stop distance (machine-discovered, NOT ATR)
    # ------------------------------------------------------------------

    def identify_stop_level(
        self, trade_record: dict, price_history: pd.DataFrame
    ) -> float:
        """For TYPE_A trades: compute optimal stop distance from price data.

        Analyzes the distribution of favorable excursions before reversals.
        Calculates the stop distance that would have preserved the win or minimized the loss.

        THIS IS NOT ATR — it's the machine's own discovered optimal stop distance.

        Returns recommended stop distance as a percentage of entry price.
        """
        entry_price = trade_record.get("entry_price", 0)
        direction = trade_record.get("direction", "")
        entry_time = trade_record.get("entry_time")
        exit_time = trade_record.get("exit_time")

        if entry_price == 0:
            return 0.02  # Default 2%

        # Get price data during the trade
        trade_prices = np.array([])
        if price_history is not None and "timestamp" in price_history.columns:
            if isinstance(entry_time, str):
                entry_time = pd.Timestamp(entry_time)
            if isinstance(exit_time, str):
                exit_time = pd.Timestamp(exit_time)

            mask = (
                (price_history["timestamp"] >= entry_time) &
                (price_history["timestamp"] <= exit_time)
            )
            trade_prices = price_history.loc[mask, "close"].values

        if len(trade_prices) < 2:
            return 0.02

        # Calculate distribution of adverse excursions
        adverse_excursions = []
        for i in range(1, len(trade_prices)):
            if direction == "LONG":
                excursion = (trade_prices[i] - entry_price) / entry_price
            else:
                excursion = (entry_price - trade_prices[i]) / entry_price
            adverse_excursions.append(excursion)

        adverse_excursions = np.array(adverse_excursions)

        if len(adverse_excursions) == 0:
            return 0.02

        # The optimal stop is at the 75th percentile of adverse excursions
        # (covers 75% of adverse moves while avoiding the worst 25%)
        optimal_stop = np.percentile(np.abs(adverse_excursions), 75)

        # Clamp to reasonable bounds
        optimal_stop = max(0.005, min(optimal_stop, 0.05))  # Between 0.5% and 5%

        return float(optimal_stop)

    # ------------------------------------------------------------------
    # Forensics report
    # ------------------------------------------------------------------

    def generate_forensics_report(
        self, trades_list: List[dict], price_data: pd.DataFrame
    ) -> Tuple[dict, str]:
        """Generate comprehensive forensics report.

        Returns (report_dict, formatted_string).
        """
        df = self.analyze_batch(trades_list, price_data)

        if len(df) == 0:
            empty_report = {
                "total_trades_analyzed": 0,
                "classification_counts": {},
                "key_findings": {},
                "recommendations": [],
            }
            return empty_report, "No trades to analyze."

        classification_counts = df["classification"].value_counts().to_dict()

        # Key findings per classification
        key_findings = {}
        for cls_type in ["TYPE_A", "TYPE_B", "TYPE_C", "TYPE_D"]:
            cls_trades = df[df["classification"] == cls_type]
            if len(cls_trades) > 0:
                key_findings[f"TYPE_{cls_type[-1]}"] = {
                    "count": int(len(cls_trades)),
                    "pct": round(len(cls_trades) / len(df) * 100, 1),
                    "recommendation": cls_trades.iloc[0]["recommended_action"],
                }

        # Recommendations
        recommendations = []
        type_a_count = classification_counts.get("TYPE_A", 0)
        type_b_count = classification_counts.get("TYPE_B", 0)
        type_c_count = classification_counts.get("TYPE_C", 0)
        type_d_count = classification_counts.get("TYPE_D", 0)

        if type_a_count > 0:
            # Calculate optimal stops for TYPE_A trades
            optimal_stops = []
            for _, row in df[df["classification"] == "TYPE_A"].iterrows():
                details = json.loads(row["details_json"])
                trade_rec = {
                    "direction": details["direction"],
                    "entry_price": details["entry_price"],
                    "exit_price": details["exit_price"],
                }
                # We pass the full record but need a minimal dict for identify_stop_level
                rec = {
                    "entry_price": details["entry_price"],
                    "direction": details["direction"],
                }
                try:
                    stop = self.identify_stop_level(rec, price_data)
                    optimal_stops.append(stop)
                except Exception:
                    pass

            if optimal_stops:
                avg_stop = np.mean(optimal_stops)
                recommendations.append({
                    "type": "TYPE_A",
                    "action": "Widen stop distance",
                    "current_stop_pct": round(avg_stop * 100, 2),
                    "reason": f"Based on {len(optimal_stops)} losing trades where price moved favorably before reversing",
                })

        if type_b_count > 0:
            recommendations.append({
                "type": "TYPE_B",
                "action": "Retrain on entry patterns",
                "reason": f"{type_b_count} trades never moved favorably — entry logic needs revision",
            })

        if type_c_count > 0:
            recommendations.append({
                "type": "TYPE_C",
                "action": "Increase regime detection sensitivity",
                "reason": f"{type_c_count} trades failed due to regime shift — add market state filter",
            })

        if type_d_count > 0:
            recommendations.append({
                "type": "TYPE_D",
                "action": "Increase minimum signal confidence threshold",
                "reason": f"{type_d_count} trades were noise — filter low-confidence signals",
            })

        report = {
            "total_trades_analyzed": len(df),
            "classification_counts": classification_counts,
            "key_findings": key_findings,
            "recommendations": recommendations,
            "latent_clusters": [],
        }

        # Format as string
        lines = [
            f"=" * 60,
            f"LOSS FORENSICS REPORT — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"=" * 60,
            f"",
            f"Total trades analyzed: {len(df)}",
            f"",
            f"Classification breakdown:",
            f"  TYPE_A (stop too tight):  {type_a_count} ({type_a_count/max(len(df),1)*100:.1f}%)",
            f"  TYPE_B (wrong direction): {type_b_count} ({type_b_count/max(len(df),1)*100:.1f}%)",
            f"  TYPE_C (regime change):   {type_c_count} ({type_c_count/max(len(df),1)*100:.1f}%)",
            f"  TYPE_D (noise/whipsaw):   {type_d_count} ({type_d_count/max(len(df),1)*100:.1f}%)",
            f"",
            f"Recommended parameter adjustments:",
        ]

        for rec in recommendations:
            lines.append(f"  [{rec['type']}] {rec['action']}: {rec['reason']}")

        lines.append("")
        lines.append("=" * 60)

        formatted = "\n".join(lines)

        return report, formatted

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store_forensics(self, ticker: str, report: dict) -> int:
        """Save forensics report to the loss forensics database."""
        if self.storage is None:
            self.logger.warning("No storage backend configured for forensics")
            return -1

        try:
            report_id = self.storage.save_loss_forensics(ticker, report)
            self.logger.info(f"Stored forensics report for {ticker} (ID: {report_id})")
            return report_id
        except Exception as e:
            self.logger.error(f"Failed to store forensics: {e}")
            return -1
