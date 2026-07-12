#!/usr/bin/env python3
"""
train_all.py — End-to-end training pipeline for the autonomous trading machine.

1. Fetches all historical 1-minute data for all 9 tickers (2018-01-01 to 2026-07-10)
2. Validates and cleans all data
3. Trains world model + RL agent for each ticker (SPY first)
4. Runs full backtest from 2023-01-01 to 2026-07-10
5. Prints backtest summary with win rate, net profit, Sharpe, max drawdown
6. Saves all results

Usage:
    export FMP_API_KEY=your_key_here
    python train_all.py
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.config import get_config, reset_config
from utils.logger import setup_logger, get_logger

# ============================================================================
# Step 0: Initialize
# ============================================================================

setup_logger()
logger = get_logger()
config = reset_config()

TICKER_ORDER = ["SPY", "TSLA", "NVDA", "AMZN", "NFLX", "CSCO", "QQQ", "IWM", "SPX"]

logger.info("=" * 70)
logger.info("AUTONOMOUS TRADING MACHINE — FULL TRAINING PIPELINE")
logger.info("=" * 70)
logger.info(f"Tickers: {TICKER_ORDER}")
logger.info(f"Full data range: {config.START_DATE} to {config.END_DATE}")
logger.info(f"Backtest range: 2023-01-01 to 2026-07-10")
logger.info(f"Model dir: {config.MODEL_DIR}")
logger.info(f"Data dir: {config.DATA_DIR}")
logger.info(f"Device: {'cuda' if __import__('torch').cuda.is_available() else 'mps' if __import__('torch').backends.mps.is_available() else 'cpu'}")

# ============================================================================
# Step 1: Fetch all historical data
# ============================================================================

def step_fetch_all_data():
    """Fetch 1-minute intraday data for all tickers from 2018-01-01 to 2026-07-10."""
    logger.info("=" * 70)
    logger.info("STEP 1: FETCHING HISTORICAL DATA")
    logger.info("=" * 70)

    api_key = config.get_api_key()
    if not api_key:
        logger.error("FMP_API_KEY not set. Export it or add to .env file.")
        logger.error("  export FMP_API_KEY=your_key_here")
        sys.exit(1)

    from data.fetcher import FMPDataFetcher, DataFetchError
    from data.validation import DataValidator
    from data.storage import DataStorage

    fetcher = FMPDataFetcher(api_key)
    validator = DataValidator()
    storage = DataStorage(config.DATA_DIR)

    results = {}
    total_ticks = 0

    for ticker in TICKER_ORDER:
        logger.info(f"  Fetching {ticker}...")
        try:
            df = fetcher.fetch_historical_intraday(
                ticker, config.START_DATE, config.END_DATE
            )
            if len(df) == 0:
                logger.warning(f"  {ticker}: No intraday data, trying daily fallback...")
                df = fetcher.fetch_historical_daily(
                    ticker, config.START_DATE, config.END_DATE
                )

            if len(df) > 0:
                # Validate
                is_valid, issues = validator.validate_tick_data(df)
                if not is_valid:
                    logger.warning(f"  {ticker}: Validation issues: {issues}")
                    df = validator.clean_tick_data(df)

                # Check completeness
                completeness = validator.check_data_completeness(
                    ticker, config.START_DATE, config.END_DATE
                )

                # Save
                storage.save_tick_data(ticker, df)
                results[ticker] = df
                total_ticks += len(df)

                logger.info(
                    f"  {ticker}: {len(df):,} ticks saved "
                    f"({completeness:.1f}% complete)"
                )
            else:
                logger.error(f"  {ticker}: No data available, skipping")
        except DataFetchError as e:
            logger.error(f"  {ticker}: Fetch failed: {e}")
        except Exception as e:
            logger.error(f"  {ticker}: Unexpected error: {e}")

    logger.info(f"  Fetch complete. {len(results)}/{len(TICKER_ORDER)} tickers, {total_ticks:,} total ticks")
    return results

# ============================================================================
# Step 2: Train models for each ticker
# ============================================================================

def step_train_all_models(fetched_data):
    """Train world model and RL agent for each ticker in order."""
    logger.info("=" * 70)
    logger.info("STEP 2: TRAINING MODELS")
    logger.info("=" * 70)

    from data.storage import DataStorage, DataNotFoundError
    from models.ticker_manager import TickerManager
    from models.world_model import PriceVAE

    storage = DataStorage(config.DATA_DIR)
    ticker_mgr = TickerManager(config)

    training_results = {}

    for ticker in TICKER_ORDER:
        logger.info("")
        logger.info(f"  --- {ticker} ---")

        # Load prices
        try:
            prices = storage.get_close_prices(ticker)
            logger.info(f"  {ticker}: {len(prices):,} close prices loaded")
        except DataNotFoundError:
            logger.error(f"  {ticker}: No stored data. Skipping.")
            continue

        if len(prices) < 700:
            logger.error(
                f"  {ticker}: Only {len(prices)} ticks (need 700+). Skipping."
            )
            continue

        # ---- Train World Model ----
        logger.info(f"  Training World Model for {ticker}...")
        try:
            ticker_mgr.initialize_ticker(ticker)
            wm_history = ticker_mgr.train_world_model(
                ticker, prices,
                epochs=config.PPO_EPOCHS * 10,  # 100 epochs default
            )

            best_val = wm_history.get("best_val_loss", float("inf"))
            best_epoch = wm_history.get("best_epoch", 0)
            logger.info(
                f"  {ticker}: World model trained — "
                f"best val loss: {best_val:.6f} (epoch {best_epoch})"
            )
        except Exception as e:
            logger.error(f"  {ticker}: World model training failed: {e}")
            continue

        # ---- Train RL Agent ----
        logger.info(f"  Training RL Agent for {ticker}...")
        try:
            rl_history = ticker_mgr.train_rl_agent(
                ticker, prices,
                timesteps=config.PPO_STEPS_PER_EPOCH * config.PPO_EPOCHS,
            )
            logger.info(f"  {ticker}: RL agent trained")
        except Exception as e:
            logger.error(f"  {ticker}: RL agent training failed: {e}")
            # Continue anyway — world model is still useful

        version = ticker_mgr._registry[ticker]["version"]
        logger.info(f"  {ticker} training complete. Version: {version}")

        training_results[ticker] = {
            "version": version,
            "world_model_val_loss": wm_history.get("best_val_loss", float("inf")),
            "prices_count": len(prices),
        }

    # Save all models
    ticker_mgr.save_all_models()
    logger.info("  All models saved.")

    return ticker_mgr, training_results

# ============================================================================
# Step 3: Run backtest
# ============================================================================

def step_run_backtest(ticker_mgr, training_results):
    """Run walk-forward backtest from 2023-01-01 to 2026-07-10 on all trained tickers."""
    logger.info("=" * 70)
    logger.info("STEP 3: BACKTEST (2023-01-01 to 2026-07-10)")
    logger.info("=" * 70)

    from data.storage import DataStorage
    from evolution.backtester import Backtester
    from evolution.loss_forensics import LossForensics
    from evolution.auto_adapt import AutoAdapter

    storage = DataStorage(config.DATA_DIR)
    backtester = Backtester(storage=storage)

    backtest_results = {}
    aggregate = {
        "total_trades": 0,
        "total_wins": 0,
        "total_losses": 0,
        "total_net_profit": 0.0,
        "sharpe_ratios": [],
        "max_drawdowns": [],
    }

    for ticker in TICKER_ORDER:
        if not ticker_mgr.is_ticker_ready(ticker):
            logger.warning(f"  {ticker}: Not ready, skipping backtest")
            continue

        try:
            prices = storage.get_close_prices(ticker, "2023-01-01", "2026-07-10")
            if len(prices) < 500:
                logger.warning(f"  {ticker}: Only {len(prices)} ticks in backtest range, skipping")
                continue

            logger.info(f"  Backtesting {ticker} ({len(prices):,} ticks)...")
            result = backtester.run_backtest(ticker, prices, ticker_manager=ticker_mgr)

            backtest_results[ticker] = result

            aggregate["total_trades"] += result["total_trades"]
            aggregate["total_wins"] += result["total_wins"]
            aggregate["total_losses"] += result["total_losses"]
            aggregate["total_net_profit"] += result["net_profit"]
            if result["sharpe_ratio"] != 0:
                aggregate["sharpe_ratios"].append(result["sharpe_ratio"])
            aggregate["max_drawdowns"].append(result["max_drawdown_pct"])

            logger.info(
                f"  {ticker}: {result['total_trades']} trades | "
                f"Win: {result['win_rate']*100:.1f}% | "
                f"P&L: ${result['net_profit']:,.2f} | "
                f"Sharpe: {result['sharpe_ratio']:.2f} | "
                f"Max DD: {result['max_drawdown_pct']*100:.1f}%"
            )

            # Run loss forensics
            try:
                trades_df = storage.load_trades(ticker=ticker)
                if len(trades_df) > 0:
                    losing_trades = trades_df[trades_df["pnl"] < 0].to_dict("records")
                    if losing_trades:
                        world_model = ticker_mgr._registry[ticker]["world_model"]
                        forensics = LossForensics(storage=storage, world_model=world_model)
                        tick_data = storage.load_tick_data(ticker)
                        report, formatted = forensics.generate_forensics_report(
                            losing_trades, tick_data
                        )
                        forensics.store_forensics(ticker, report)

                        # Auto-adapt
                        adapter = AutoAdapter(ticker_manager=ticker_mgr, storage=storage)
                        changes = adapter.apply_fixes(ticker, report)
                        if changes["changes"]:
                            logger.info(f"  {ticker}: Auto-adapted {len(changes['changes'])} parameters")
            except Exception as e:
                logger.warning(f"  {ticker}: Forensics skipped: {e}")

        except Exception as e:
            logger.error(f"  {ticker}: Backtest failed: {e}")

    # ========================================================================
    # Print backtest summary
    # ========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("BACKTEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Date range:        2023-01-01 to 2026-07-10")
    logger.info(f"  Tickers tested:    {len(backtest_results)}/{len(TICKER_ORDER)}")
    logger.info(f"  Total trades:      {aggregate['total_trades']}")
    logger.info(f"  Total wins:        {aggregate['total_wins']}")
    logger.info(f"  Total losses:      {aggregate['total_losses']}")

    if aggregate["total_trades"] > 0:
        win_rate = aggregate["total_wins"] / aggregate["total_trades"] * 100
    else:
        win_rate = 0.0
    logger.info(f"  Win rate:          {win_rate:.1f}%")

    net_profit = aggregate["total_net_profit"]
    logger.info(f"  Net profit:        ${net_profit:,.2f}")

    if aggregate["sharpe_ratios"]:
        avg_sharpe = np.mean(aggregate["sharpe_ratios"])
    else:
        avg_sharpe = 0.0
    logger.info(f"  Avg Sharpe ratio:  {avg_sharpe:.2f}")

    if aggregate["max_drawdowns"]:
        max_dd = max(aggregate["max_drawdowns"]) * 100
    else:
        max_dd = 0.0
    logger.info(f"  Max drawdown:      {max_dd:.1f}%")
    logger.info("=" * 70)

    # Per-ticker detail
    logger.info("")
    logger.info("PER-TICKER RESULTS:")
    logger.info("-" * 70)
    logger.info(f"  {'Ticker':<8} {'Trades':>7} {'Win Rate':>9} {'Net P&L':>12} {'Sharpe':>8} {'Max DD':>8}")
    logger.info("-" * 70)
    for ticker in TICKER_ORDER:
        if ticker in backtest_results:
            r = backtest_results[ticker]
            logger.info(
                f"  {ticker:<8} {r['total_trades']:>7} "
                f"{r['win_rate']*100:>8.1f}% "
                f"${r['net_profit']:>11,.2f} "
                f"{r['sharpe_ratio']:>8.2f} "
                f"{r['max_drawdown_pct']*100:>7.1f}%"
            )
        else:
            logger.info(f"  {ticker:<8} {'—':>7} {'—':>9} {'—':>12} {'—':>8} {'—':>8}")
    logger.info("-" * 70)

    return backtest_results

# ============================================================================
# Step 4: Save all results
# ============================================================================

def step_save_results(training_results, backtest_results):
    """Save training and backtest results to disk."""
    logger.info("=" * 70)
    logger.info("STEP 4: SAVING RESULTS")
    logger.info("=" * 70)

    results_dir = Path(config.MODEL_DIR) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Training summary CSV
    if training_results:
        train_df = pd.DataFrame(training_results).T
        train_df.index.name = "ticker"
        train_path = results_dir / f"training_summary_{timestamp}.csv"
        train_df.to_csv(train_path)
        logger.info(f"  Training summary: {train_path}")

    # Backtest summary CSV
    if backtest_results:
        bt_df = pd.DataFrame(backtest_results).T
        bt_df.index.name = "ticker"
        bt_path = results_dir / f"backtest_summary_{timestamp}.csv"
        bt_df.to_csv(bt_path)
        logger.info(f"  Backtest summary: {bt_path}")

    # Also export via the Exports utility
    try:
        from dashboard.exports import Exports
        exports = Exports()
        export_dir = results_dir / f"full_export_{timestamp}"
        exported = exports.export_all(str(export_dir))
        for data_type, filepath in exported.items():
            logger.info(f"  Exported {data_type}: {filepath}")
    except Exception as e:
        logger.warning(f"  Export utility skipped: {e}")

    logger.info("  All results saved.")

# ============================================================================
# Main
# ============================================================================

def main():
    start_time = time.time()

    # Step 1: Fetch data
    fetched_data = step_fetch_all_data()

    # Step 2: Train models
    ticker_mgr, training_results = step_train_all_models(fetched_data)

    # Step 3: Backtest
    backtest_results = step_run_backtest(ticker_mgr, training_results)

    # Step 4: Save
    step_save_results(training_results, backtest_results)

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"TRAINING PIPELINE COMPLETE — {hours}h {minutes}m {seconds}s")
    logger.info("=" * 70)
    logger.info("")
    logger.info("  Next steps:")
    logger.info("    python run.py dashboard    # Launch the Streamlit dashboard")
    logger.info("    python run.py live          # Start live trading mode")
    logger.info("    python run.py status        # Check system status")
    logger.info("")


if __name__ == "__main__":
    main()
