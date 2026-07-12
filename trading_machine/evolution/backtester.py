"""
evolution/backtester.py — Walk-forward backtesting engine with realistic fills.

Full implementation:
- Walk-forward backtesting on historical data
- Realistic trade simulation (slippage, commission, latency)
- Complete performance metrics (Sharpe, Sortino, Calmar, profit factor)
- Model version comparison (baseline vs proposed)
- Integration with loss forensics for automatic post-backtest analysis
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from utils.config import get_config
from utils.logger import get_logger


class Backtester:
    """Walk-forward backtesting engine with realistic fills and forensics integration.

    Simulates trading with:
    - Slippage: configurable basis points per trade
    - Commission: configurable per-trade fee
    - Latency: minimum bars between signal and execution
    - Max drawdown enforcement: auto-liquidate on breach
    """

    def __init__(
        self,
        storage=None,
        slippage_bps: float = 1.0,
        commission_per_trade: float = 0.0,
        latency_bars: int = 1,
    ):
        self.storage = storage
        self.config = get_config()
        self.logger = get_logger()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        # Realistic fill parameters
        self.slippage_bps = slippage_bps          # Basis points per trade
        self.commission_per_trade = commission_per_trade
        self.latency_bars = latency_bars           # Bars between signal and fill

        # Store version comparison data
        self._version_results: Dict[str, List[dict]] = {}

    # ======================================================================
    # run_backtest — Main backtesting method
    # ======================================================================

    def run_backtest(
        self,
        ticker: str,
        prices: np.ndarray,
        ticker_manager=None,
        window_size: int = 500,
        step_size: int = 100,
        model_version: str = "",
    ) -> dict:
        """Run walk-forward backtest on historical price data.

        Simulates realistic trading with slippage, commission, and
        latency between signal generation and fill execution.

        Args:
            ticker: Ticker symbol.
            prices: 1D numpy array of close prices.
            ticker_manager: TickerManager for getting trading signals.
            window_size: Ticks in the lookback window.
            step_size: Ticks to advance per step.
            model_version: Version label for the model being tested.

        Returns:
            Dict with all backtest metrics.
        """
        if len(prices) < window_size + 50:
            self.logger.error(f"Not enough data for backtest: {len(prices)} ticks")
            return self._empty_result(ticker, model_version)

        total_ticks = len(prices)
        num_steps = (total_ticks - window_size) // step_size

        self.logger.info(
            f"Backtesting {ticker}: {total_ticks} ticks, "
            f"{num_steps} steps, window={window_size}, "
            f"slippage={self.slippage_bps}bps, latency={self.latency_bars}bar"
        )

        # State
        position = 0
        entry_price = 0.0
        entry_step = 0
        trades: List[dict] = []
        equity_curve = [self.config.INITIAL_CAPITAL]
        equity = self.config.INITIAL_CAPITAL
        peak_equity = equity
        max_drawdown = 0.0
        pending_signal = None
        pending_countdown = 0

        for step in range(num_steps):
            start_idx = step * step_size
            end_idx = start_idx + window_size
            if end_idx >= total_ticks:
                break

            price_window = prices[start_idx:end_idx]
            current_price = prices[end_idx - 1]

            # Process pending signal (latency simulation)
            if pending_countdown > 0:
                pending_countdown -= 1
                if pending_countdown == 0 and pending_signal is not None:
                    action = pending_signal
                    pending_signal = None
                else:
                    action = 2  # Hold while waiting
            else:
                # Get signal from model
                if ticker_manager is not None:
                    try:
                        latent = ticker_manager.get_latent_state(ticker, price_window)
                        action, confidence = ticker_manager.get_signal(ticker, latent)

                        # Apply latency
                        if self.latency_bars > 0:
                            pending_signal = action
                            pending_countdown = self.latency_bars
                            action = 2  # Hold until latency expires
                    except Exception as e:
                        self.logger.warning(f"Signal error at step {step}: {e}")
                        action = 2
                else:
                    action = 2

            # Execute action with realistic fill
            fill_price = self._apply_slippage(current_price, action)
            fill_price, fill_cost = self._apply_commission(fill_price)

            if action == 0 and position <= 0:  # LONG
                if position == -1:
                    pnl_pct = (entry_price - fill_price) / entry_price
                    pnl_dollar = equity * pnl_pct - fill_cost
                    equity += pnl_dollar
                    trades.append({
                        "direction": "SHORT",
                        "entry_price": entry_price,
                        "exit_price": fill_price,
                        "pnl": pnl_dollar,
                        "pnl_pct": pnl_pct,
                        "step": step,
                        "entry_step": entry_step,
                        "exit_step": step,
                        "slippage_cost": abs(current_price - fill_price),
                        "commission": fill_cost,
                    })
                position = 1
                entry_price = fill_price
                entry_step = step

            elif action == 1 and position >= 0:  # SHORT
                if position == 1:
                    pnl_pct = (fill_price - entry_price) / entry_price
                    pnl_dollar = equity * pnl_pct - fill_cost
                    equity += pnl_dollar
                    trades.append({
                        "direction": "LONG",
                        "entry_price": entry_price,
                        "exit_price": fill_price,
                        "pnl": pnl_dollar,
                        "pnl_pct": pnl_pct,
                        "step": step,
                        "entry_step": entry_step,
                        "exit_step": step,
                        "slippage_cost": abs(current_price - fill_price),
                        "commission": fill_cost,
                    })
                position = -1
                entry_price = fill_price
                entry_step = step

            elif action == 2 and position != 0:  # EXIT
                pnl_pct = (
                    (fill_price - entry_price) / entry_price if position == 1
                    else (entry_price - fill_price) / entry_price
                )
                pnl_dollar = equity * pnl_pct - fill_cost
                equity += pnl_dollar
                trades.append({
                    "direction": "LONG" if position == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": fill_price,
                    "pnl": pnl_dollar,
                    "pnl_pct": pnl_pct,
                    "step": step,
                    "entry_step": entry_step,
                    "exit_step": step,
                    "slippage_cost": abs(current_price - fill_price),
                    "commission": fill_cost,
                })
                position = 0
                entry_price = 0.0
                entry_step = 0

            # Track equity and drawdown
            unrealized = 0.0
            if position == 1 and entry_price > 0:
                unrealized = equity * (current_price - entry_price) / entry_price
            elif position == -1 and entry_price > 0:
                unrealized = equity * (entry_price - current_price) / entry_price

            total_equity = equity + unrealized
            equity_curve.append(total_equity)

            if total_equity > peak_equity:
                peak_equity = total_equity

            dd = (peak_equity - total_equity) / max(peak_equity, 1.0)
            if dd > max_drawdown:
                max_drawdown = dd

            # Max drawdown enforcement
            if dd >= self.config.MAX_DRAWDOWN_PCT:
                self.logger.warning(
                    f"Max drawdown {dd*100:.1f}% exceeded at step {step}. Liquidating."
                )
                if position != 0:
                    pnl_pct = (
                        (current_price - entry_price) / entry_price if position == 1
                        else (entry_price - current_price) / entry_price
                    )
                    pnl_dollar = equity * pnl_pct
                    equity += pnl_dollar
                    trades.append({
                        "direction": "LONG" if position == 1 else "SHORT",
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "pnl": pnl_dollar,
                        "pnl_pct": pnl_pct,
                        "step": step,
                        "entry_step": entry_step,
                        "exit_step": step,
                        "exit_reason": "max_drawdown",
                        "slippage_cost": 0.0,
                        "commission": 0.0,
                    })
                    position = 0
                break

        # Close open position at end
        if position != 0 and len(prices) > 0:
            final_price = prices[-1]
            pnl_pct = (
                (final_price - entry_price) / entry_price if position == 1
                else (entry_price - final_price) / entry_price
            )
            pnl_dollar = equity * pnl_pct
            equity += pnl_dollar
            trades.append({
                "direction": "LONG" if position == 1 else "SHORT",
                "entry_price": entry_price,
                "exit_price": final_price,
                "pnl": pnl_dollar,
                "pnl_pct": pnl_pct,
                "step": num_steps,
                "entry_step": entry_step,
                "exit_step": num_steps,
                "exit_reason": "end_of_data",
                "slippage_cost": 0.0,
                "commission": 0.0,
            })

        # Compute all metrics
        result = self._compute_all_metrics(ticker, trades, equity_curve, max_drawdown, model_version)

        # Store for version comparison
        if model_version:
            if model_version not in self._version_results:
                self._version_results[model_version] = []
            self._version_results[model_version].append(result)

        # Save to storage
        if self.storage is not None:
            try:
                self.storage.save_backtest_result(result)
            except Exception as e:
                self.logger.error(f"Failed to save backtest result: {e}")

        return result

    # ------------------------------------------------------------------
    # Realistic fills
    # ------------------------------------------------------------------

    def _apply_slippage(self, price: float, action: int) -> float:
        """Apply slippage to the fill price.

        For LONG entries and SHORT exits: price moves against you (higher).
        For SHORT entries and LONG exits: price moves against you (lower).
        """
        if self.slippage_bps <= 0:
            return price

        slip = price * (self.slippage_bps / 10000.0)
        # Add random component: 0.5x to 1.5x of base slippage
        slip *= 0.5 + np.random.random()

        if action == 0:  # Entering LONG — price worse (higher)
            return price + slip
        elif action == 1:  # Entering SHORT — price worse (lower)
            return price - slip
        else:
            return price

    def _apply_commission(self, price: float) -> Tuple[float, float]:
        """Apply per-trade commission. Returns (adjusted_price, commission_cost)."""
        if self.commission_per_trade <= 0:
            return price, 0.0
        return price, self.commission_per_trade

    # ======================================================================
    # Metrics computation
    # ======================================================================

    def _compute_all_metrics(
        self,
        ticker: str,
        trades: List[dict],
        equity_curve: List[float],
        max_drawdown: float,
        model_version: str = "",
    ) -> dict:
        """Compute comprehensive performance metrics from trade list.

        Includes: win rate, profit factor, Sharpe, Sortino, Calmar,
        expectancy, average holding period, and more.
        """
        if not trades:
            return self._empty_result(ticker, model_version)

        pnls = np.array([t["pnl"] for t in trades])
        pnl_pcts = np.array([t.get("pnl_pct", 0) for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        total_trades = len(trades)
        total_wins = len(wins)
        total_losses = len(losses)
        win_rate = total_wins / total_trades if total_trades > 0 else 0.0
        loss_rate = total_losses / total_trades if total_trades > 0 else 0.0

        total_win_amount = float(np.sum(wins)) if len(wins) > 0 else 0.0
        total_loss_amount = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
        net_profit = float(np.sum(pnls))

        profit_factor = (
            total_win_amount / total_loss_amount if total_loss_amount > 0
            else float("inf") if total_win_amount > 0 else 0.0
        )

        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(abs(np.mean(losses))) if len(losses) > 0 else 0.0
        largest_win = float(np.max(wins)) if len(wins) > 0 else 0.0
        largest_loss = float(np.min(pnls)) if len(pnls) > 0 else 0.0

        # Expectancy
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

        # Average holding period (in steps)
        holding_periods = []
        for t in trades:
            if "entry_step" in t and "exit_step" in t:
                holding_periods.append(t["exit_step"] - t["entry_step"])
        avg_holding = float(np.mean(holding_periods)) if holding_periods else 0.0

        # Sharpe ratio (annualized, 390 min/day * 252 days)
        equity_arr = np.array(equity_curve)
        if len(equity_arr) > 1:
            returns = np.diff(equity_arr) / np.clip(equity_arr[:-1], 1e-8, None)
            returns = returns[np.isfinite(returns)]
            if len(returns) > 1 and returns.std() > 0:
                sharpe = np.sqrt(252 * 390) * (returns.mean() / returns.std())
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # Sortino ratio (only penalizes downside volatility)
        if len(equity_arr) > 1:
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 1 and downside_returns.std() > 0:
                sortino = np.sqrt(252 * 390) * (returns.mean() / downside_returns.std())
            else:
                sortino = 0.0 if len(downside_returns) == 0 else sharpe
        else:
            sortino = 0.0

        # Calmar ratio (annualized return / max drawdown)
        if max_drawdown > 0 and len(equity_arr) > 1:
            total_return = (equity_arr[-1] - equity_arr[0]) / equity_arr[0]
            annualized_return = total_return  # Simplified
            calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
        else:
            calmar = 0.0

        # Consecutive wins/losses
        consec_wins = 0
        consec_losses = 0
        max_consec_wins = 0
        max_consec_losses = 0
        for p in pnls:
            if p > 0:
                consec_wins += 1
                consec_losses = 0
                max_consec_wins = max(max_consec_wins, consec_wins)
            elif p < 0:
                consec_losses += 1
                consec_wins = 0
                max_consec_losses = max(max_consec_losses, consec_losses)

        # Total slippage and commission costs
        total_slippage = sum(t.get("slippage_cost", 0.0) for t in trades)
        total_commission = sum(t.get("commission", 0.0) for t in trades)

        result = {
            "ticker": ticker,
            "run_date": datetime.utcnow(),
            "model_version": model_version,
            # Core metrics
            "win_rate": round(win_rate, 4),
            "loss_rate": round(loss_rate, 4),
            "total_trades": total_trades,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "total_win_amount": round(total_win_amount, 2),
            "total_loss_amount": round(total_loss_amount, 2),
            "net_profit": round(net_profit, 2),
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else 9999.0,
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "largest_win": round(largest_win, 2),
            "largest_loss": round(largest_loss, 2),
            "max_drawdown_pct": round(max_drawdown, 4),
            "sharpe_ratio": round(sharpe, 4),
            # Extended metrics
            "sortino_ratio": round(sortino, 4),
            "calmar_ratio": round(calmar, 4),
            "expectancy": round(expectancy, 4),
            "avg_holding_bars": round(avg_holding, 1),
            "max_consec_wins": max_consec_wins,
            "max_consec_losses": max_consec_losses,
            "total_slippage": round(total_slippage, 2),
            "total_commission": round(total_commission, 2),
        }

        self.logger.info(
            f"Backtest {ticker}: {total_trades} trades | "
            f"Win: {win_rate:.1%} | Net: ${net_profit:,.2f} | "
            f"Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f} | "
            f"Max DD: {max_drawdown:.1%}"
        )

        return result

    # ======================================================================
    # Model version comparison
    # ======================================================================

    def compare_versions(
        self, version_a: str, version_b: str, ticker: Optional[str] = None
    ) -> dict:
        """Compare backtest results between two model versions.

        Args:
            version_a: Baseline version label.
            version_b: Proposed version label.
            ticker: Optional ticker filter.

        Returns:
            Comparison dict with deltas for each metric.
        """
        results_a = self._version_results.get(version_a, [])
        results_b = self._version_results.get(version_b, [])

        if ticker:
            results_a = [r for r in results_a if r["ticker"] == ticker]
            results_b = [r for r in results_b if r["ticker"] == ticker]

        if not results_a or not results_b:
            return {"error": "No results for one or both versions", "version_a": version_a, "version_b": version_b}

        metrics = [
            "win_rate", "net_profit", "profit_factor", "sharpe_ratio",
            "sortino_ratio", "calmar_ratio", "max_drawdown_pct",
            "expectancy", "avg_win", "avg_loss",
        ]

        comparison = {"version_a": version_a, "version_b": version_b, "deltas": {}}

        for metric in metrics:
            val_a = np.mean([r[metric] for r in results_a])
            val_b = np.mean([r[metric] for r in results_b])
            delta = val_b - val_a
            pct_change = (delta / max(abs(val_a), 1e-8)) * 100
            comparison["deltas"][metric] = {
                "baseline": round(val_a, 4),
                "proposed": round(val_b, 4),
                "delta": round(delta, 4),
                "pct_change": round(pct_change, 2),
            }

        # Overall assessment
        profit_delta = comparison["deltas"].get("net_profit", {}).get("delta", 0)
        sharpe_delta = comparison["deltas"].get("sharpe_ratio", {}).get("delta", 0)
        dd_delta = comparison["deltas"].get("max_drawdown_pct", {}).get("delta", 0)

        if profit_delta > 0 and sharpe_delta > -0.1 and dd_delta < 0.05:
            comparison["recommendation"] = "PROMOTE"
        elif profit_delta > 0:
            comparison["recommendation"] = "REVIEW"
        else:
            comparison["recommendation"] = "REJECT"

        self.logger.info(
            f"Version comparison {version_a} vs {version_b}: "
            f"P&L Δ=${profit_delta:,.2f}, "
            f"Sharpe Δ={sharpe_delta:+.2f}, "
            f"→ {comparison['recommendation']}"
        )

        return comparison

    def get_version_history(self) -> Dict[str, List[dict]]:
        """Get all stored version results."""
        return dict(self._version_results)

    # ======================================================================
    # Integration with loss forensics
    # ======================================================================

    def backtest_with_forensics(
        self,
        ticker: str,
        prices: np.ndarray,
        ticker_manager=None,
        model_version: str = "",
    ) -> dict:
        """Run backtest and automatically run loss forensics on losing trades.

        Returns:
            Dict with backtest_result and forensics_report keys.
        """
        # Run backtest
        bt_result = self.run_backtest(
            ticker, prices, ticker_manager=ticker_manager,
            model_version=model_version,
        )

        # Run forensics on losing trades
        forensics_report = None
        if self.storage is not None and bt_result["total_losses"] > 0:
            try:
                from evolution.loss_forensics import LossForensics

                # Get the trades that were just generated
                trades_df = self.storage.load_trades(ticker=ticker)
                if len(trades_df) > 0:
                    losing_trades = trades_df[trades_df["pnl"] < 0].to_dict("records")
                    if losing_trades:
                        world_model = None
                        if ticker_manager and ticker in ticker_manager._registry:
                            world_model = ticker_manager._registry[ticker].get("world_model")

                        forensics = LossForensics(storage=self.storage, world_model=world_model)
                        price_data = self.storage.load_tick_data(ticker)
                        report, formatted = forensics.generate_forensics_report(
                            losing_trades, price_data
                        )
                        forensics.store_forensics(ticker, report)
                        forensics_report = report

                        self.logger.info(
                            f"Forensics: {report['total_trades_analyzed']} trades analyzed "
                            f"({bt_result['total_losses']} losses)"
                        )
            except Exception as e:
                self.logger.error(f"Forensics failed for {ticker}: {e}")

        return {
            "backtest_result": bt_result,
            "forensics_report": forensics_report,
        }

    # ======================================================================
    # Multi-ticker batch backtest
    # ======================================================================

    def run_batch_backtest(
        self,
        tickers: List[str],
        price_data: Dict[str, np.ndarray],
        ticker_manager=None,
        model_version: str = "",
    ) -> pd.DataFrame:
        """Run backtests for multiple tickers and return consolidated results.

        Args:
            tickers: List of ticker symbols.
            price_data: Dict mapping ticker to price numpy array.
            ticker_manager: TickerManager for signals.
            model_version: Version label.

        Returns:
            DataFrame with one row per ticker.
        """
        results = []

        for ticker in tickers:
            if ticker not in price_data:
                self.logger.warning(f"No price data for {ticker}, skipping")
                continue

            prices = price_data[ticker]
            if len(prices) < 500:
                self.logger.warning(f"Not enough data for {ticker}: {len(prices)} ticks")
                continue

            bt_result = self.backtest_with_forensics(
                ticker, prices, ticker_manager=ticker_manager,
                model_version=model_version,
            )

            results.append(bt_result["backtest_result"])

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df = df.sort_values("sharpe_ratio", ascending=False)

        # Print batch summary
        self.logger.info("=" * 70)
        self.logger.info("BATCH BACKTEST SUMMARY")
        self.logger.info("=" * 70)
        self.logger.info(
            f"{'Ticker':<8} {'Trades':>7} {'Win Rate':>9} "
            f"{'Net P&L':>12} {'Sharpe':>8} {'Max DD':>8}"
        )
        self.logger.info("-" * 70)
        for _, row in df.iterrows():
            self.logger.info(
                f"{row['ticker']:<8} {row['total_trades']:>7} "
                f"{row['win_rate']*100:>8.1f}% "
                f"${row['net_profit']:>11,.2f} "
                f"{row['sharpe_ratio']:>8.2f} "
                f"{row['max_drawdown_pct']*100:>7.1f}%"
            )
        self.logger.info("-" * 70)

        # Aggregate stats
        total_trades = df["total_trades"].sum()
        total_pnl = df["net_profit"].sum()
        avg_sharpe = df["sharpe_ratio"].mean()
        max_dd = df["max_drawdown_pct"].max()
        self.logger.info(
            f"TOTAL: {total_trades} trades | "
            f"Net P&L: ${total_pnl:,.2f} | "
            f"Avg Sharpe: {avg_sharpe:.2f} | "
            f"Max DD: {max_dd*100:.1f}%"
        )
        self.logger.info("=" * 70)

        return df

    # ======================================================================
    # Helpers
    # ======================================================================

    def _empty_result(self, ticker: str, model_version: str = "") -> dict:
        """Return empty result with all metrics zeroed."""
        return {
            "ticker": ticker,
            "run_date": datetime.utcnow(),
            "model_version": model_version,
            "win_rate": 0.0,
            "loss_rate": 0.0,
            "total_trades": 0,
            "total_wins": 0,
            "total_losses": 0,
            "total_win_amount": 0.0,
            "total_loss_amount": 0.0,
            "net_profit": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "expectancy": 0.0,
            "avg_holding_bars": 0.0,
            "max_consec_wins": 0,
            "max_consec_losses": 0,
            "total_slippage": 0.0,
            "total_commission": 0.0,
        }
