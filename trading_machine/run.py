#!/usr/bin/env python3
"""
run.py — CLI entry point for the autonomous trading machine.

Commands:
    fetch       Fetch historical data for all tickers
    train       Train world models and RL agents
    backtest    Run backtests on trained models
    live        Start live trading mode
    dashboard   Launch the Streamlit dashboard
    export      Export data to Excel
    status      Show system status
"""

import argparse
import os
import sys

# Ensure the project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils.config import get_config, reset_config
from utils.logger import setup_logger, get_logger


def cmd_fetch(args):
    """Fetch historical data for all tickers."""
    config = get_config()
    logger = get_logger()

    api_key = config.get_api_key()
    if not api_key:
        logger.error("FMP_API_KEY not set. Export it or add to .env file.")
        logger.error("  export FMP_API_KEY=your_key_here")
        sys.exit(1)

    from data.fetcher import FMPDataFetcher
    from data.validation import DataValidator

    logger.info("Starting data fetch...")
    fetcher = FMPDataFetcher(api_key)
    results = fetcher.fetch_all_historical_data()

    # Validate fetched data
    validator = DataValidator()
    for ticker, df in results.items():
        is_valid, issues = validator.validate_tick_data(df)
        if not is_valid:
            logger.warning(f"Data issues for {ticker}: {issues}")
        df_clean = validator.clean_tick_data(df)
        completeness = validator.check_data_completeness(
            ticker, config.START_DATE, config.END_DATE
        )
        logger.info(f"{ticker}: {len(df_clean)} rows, {completeness:.1f}% complete")

    logger.info(f"Fetch complete. Got data for {len(results)}/{len(config.TICKERS)} tickers")


def cmd_train(args):
    """Train world models and RL agents."""
    config = get_config()
    logger = get_logger()

    from data.storage import DataStorage, DataNotFoundError
    from models.ticker_manager import TickerManager

    storage = DataStorage(config.DATA_DIR)
    ticker_mgr = TickerManager(config)

    tickers = args.tickers if args.tickers else config.TICKERS

    for ticker in tickers:
        logger.info(f"=== Training {ticker} ===")
        try:
            prices = storage.get_close_prices(ticker)
            logger.info(f"Loaded {len(prices)} price points for {ticker}")

            # Train world model
            logger.info(f"Training world model for {ticker}...")
            ticker_mgr.initialize_ticker(ticker)
            wm_history = ticker_mgr.train_world_model(
                ticker, prices, epochs=args.epochs
            )
            logger.info(
                f"World model trained: best val loss = {wm_history.get('best_val_loss', 'N/A')}"
            )

            # Train RL agent
            if not args.skip_rl:
                logger.info(f"Training RL agent for {ticker}...")
                rl_history = ticker_mgr.train_rl_agent(
                    ticker, prices, timesteps=args.timesteps
                )
                logger.info(f"RL agent trained for {ticker}")

        except DataNotFoundError:
            logger.error(f"No data for {ticker}. Run 'python run.py fetch' first.")
        except Exception as e:
            logger.error(f"Failed to train {ticker}: {e}")

    # Save all models
    ticker_mgr.save_all_models()
    logger.info("All models saved.")


def cmd_backtest(args):
    """Run backtests on trained models."""
    config = get_config()
    logger = get_logger()

    from data.storage import DataStorage, DataNotFoundError
    from models.ticker_manager import TickerManager
    from evolution.backtester import Backtester
    from evolution.loss_forensics import LossForensics
    from evolution.auto_adapt import AutoAdapter

    storage = DataStorage(config.DATA_DIR)
    ticker_mgr = TickerManager(config)
    ticker_mgr.load_all_models()

    backtester = Backtester(storage=storage)

    tickers = args.tickers if args.tickers else config.TICKERS

    for ticker in tickers:
        if not ticker_mgr.is_ticker_ready(ticker):
            logger.warning(f"{ticker} not trained, skipping backtest")
            continue

        try:
            prices = storage.get_close_prices(ticker)
            logger.info(f"Backtesting {ticker} with {len(prices)} prices...")

            result = backtester.run_backtest(
                ticker, prices, ticker_manager=ticker_mgr
            )

            logger.info(
                f"{ticker}: {result['total_trades']} trades, "
                f"Win rate: {result['win_rate']*100:.1f}%, "
                f"Net P&L: ${result['net_profit']:,.2f}, "
                f"Sharpe: {result['sharpe_ratio']:.2f}"
            )

            # Run loss forensics on losing trades
            if args.forensics:
                trades = storage.load_trades(ticker=ticker)
                losing_trades = trades[trades["pnl"] < 0].to_dict("records") if len(trades) > 0 else []

                if losing_trades:
                    world_model = ticker_mgr._registry[ticker]["world_model"]
                    forensics = LossForensics(storage=storage, world_model=world_model)

                    tick_data = storage.load_tick_data(ticker)
                    report, formatted = forensics.generate_forensics_report(
                        losing_trades, tick_data
                    )

                    logger.info(f"\n{formatted}")
                    forensics.store_forensics(ticker, report)

                    # Auto-adapt
                    if args.adapt:
                        adapter = AutoAdapter(
                            ticker_manager=ticker_mgr, storage=storage
                        )
                        changes = adapter.apply_fixes(ticker, report)
                        logger.info(f"Applied {len(changes['changes'])} fixes for {ticker}")

        except DataNotFoundError:
            logger.error(f"No data for {ticker}")
        except Exception as e:
            logger.error(f"Backtest failed for {ticker}: {e}")


def cmd_live(args):
    """Start live trading mode."""
    config = get_config()
    logger = get_logger()

    api_key = config.get_api_key()
    if not api_key:
        logger.error("FMP_API_KEY not set.")
        sys.exit(1)

    logger.info("Starting live trading mode...")
    logger.info(f"Tickers: {config.TICKERS}")
    logger.info(f"Polling interval: {config.POLLING_INTERVAL_SECONDS}s")
    logger.info(f"Market hours: {config.MARKET_OPEN} - {config.MARKET_CLOSE} EST")

    from data.fetcher import FMPDataFetcher
    from data.storage import DataStorage
    from models.ticker_manager import TickerManager
    from live.screener import Screener
    from live.position_manager import PositionManager
    from live.alerts import Alerts, AlertLevel
    from utils.scheduler import get_scheduler

    # Initialize components
    storage = DataStorage(config.DATA_DIR)
    fetcher = FMPDataFetcher(api_key)
    ticker_mgr = TickerManager(config)
    ticker_mgr.load_all_models()

    screener = Screener(ticker_manager=ticker_mgr, fetcher=fetcher, storage=storage)
    position_mgr = PositionManager(storage=storage)
    alerts = Alerts()

    # Pre-fill price windows
    ready = screener.initialize_windows()
    logger.info(f"Initialized {ready}/{len(config.TICKERS)} ticker windows")

    # Define polling function
    def polling_cycle():
        signals = screener.screen_all()
        if signals:
            for signal in signals[:5]:  # Log top 5
                logger.info(
                    f"[{signal['ticker']}] {signal['signal_label']} "
                    f"(conf: {signal['confidence']:.2%}, price: ${signal['price']:.2f})"
                )
            # Alert on top signal
            top = signals[0]
            if top["signal"] != Screener.SIGNAL_NEUTRAL:
                alerts.signal_alert(
                    top["ticker"],
                    top["signal_label"],
                    top["price"],
                    top["confidence"],
                )

        # Update position prices
        try:
            quotes = fetcher.fetch_batch_quotes(config.TICKERS)
            for ticker, quote in quotes.items():
                position_mgr.update_price(ticker, quote["price"])
        except Exception as e:
            logger.warning(f"Position update failed: {e}")

        # Check risk limits
        ok, warnings = position_mgr.check_risk_limits()
        for w in warnings:
            alerts.send(AlertLevel.WARNING, w)

        if position_mgr.is_max_drawdown_exceeded:
            alerts.max_drawdown_exceeded(position_mgr.drawdown_pct)
            try:
                quotes = fetcher.fetch_batch_quotes(config.TICKERS)
                price_map = {t: q["price"] for t, q in quotes.items()}
                position_mgr.emergency_close_all(price_map)
            except Exception:
                pass

    # Start scheduler
    scheduler_mgr = get_scheduler()
    scheduler_mgr.add_intraday_polling_job(polling_cycle)
    scheduler_mgr.add_market_open_job(lambda: logger.info("Market OPEN"))
    scheduler_mgr.add_market_close_job(lambda: logger.info("Market CLOSE"))
    scheduler_mgr.add_overnight_job(
        lambda: logger.info("Overnight processing")
    )
    scheduler_mgr.add_weekend_job(
        lambda: logger.info("Weekend maintenance — running forensics and retraining")
    )

    scheduler_mgr.start()

    logger.info("Live trading started. Press Ctrl+C to stop.")

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        scheduler_mgr.stop()
        logger.info("Live trading stopped.")


def cmd_dashboard(args):
    """Launch the Streamlit dashboard."""
    import subprocess

    dashboard_path = os.path.join(PROJECT_ROOT, "dashboard", "app.py")
    config = get_config()

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        dashboard_path,
        "--server.port", str(config.DASHBOARD_PORT),
        "--server.address", config.DASHBOARD_HOST,
    ]

    print(f"Launching dashboard at http://{config.DASHBOARD_HOST}:{config.DASHBOARD_PORT}")
    subprocess.run(cmd)


def cmd_export(args):
    """Export data to Excel."""
    config = get_config()
    logger = get_logger()

    from dashboard.exports import Exports

    export_dir = args.output or "./exports"
    exports = Exports()
    results = exports.export_all(export_dir)

    for data_type, filepath in results.items():
        logger.info(f"Exported {data_type}: {filepath}")

    logger.info(f"Export complete. {len(results)} files written to {export_dir}")


def cmd_status(args):
    """Show system status."""
    config = get_config()
    logger = get_logger()

    from data.storage import DataStorage
    from models.ticker_manager import TickerManager

    print("=" * 60)
    print("TRADING MACHINE STATUS")
    print("=" * 60)
    print(f"Tickers: {config.TICKERS}")
    print(f"Date range: {config.START_DATE} to {config.END_DATE}")
    print(f"Models dir: {config.MODEL_DIR}")
    print(f"Data dir: {config.DATA_DIR}")
    print()

    storage = DataStorage(config.DATA_DIR)
    ticker_mgr = TickerManager(config)
    ticker_mgr.load_all_models()

    print("MODEL STATUS:")
    versions = ticker_mgr.get_model_versions()
    for ticker, version in versions.items():
        ready = ticker_mgr.is_ticker_ready(ticker)
        status = "✅ READY" if ready else "❌ NOT TRAINED"
        print(f"  {ticker}: v{version} — {status}")

    print()
    print("DATA STATUS:")
    for ticker in config.TICKERS:
        try:
            date_range = storage.get_available_date_range(ticker)
            prices = storage.get_close_prices(ticker)
            print(f"  {ticker}: {len(prices)} prices, {date_range}")
        except Exception:
            print(f"  {ticker}: No data")
    print("=" * 60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Autonomous Self-Learning Trading Machine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py fetch              Fetch all historical data
  python run.py train              Train all models
  python run.py train --tickers SPY AAPL  Train specific tickers
  python run.py backtest           Run backtests
  python run.py backtest --forensics --adapt  Backtest + forensics + auto-adapt
  python run.py live               Start live trading
  python run.py dashboard          Launch dashboard
  python run.py status             Show system status
  python run.py export             Export data to Excel
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # fetch
    subparsers.add_parser("fetch", help="Fetch historical data for all tickers")

    # train
    train_parser = subparsers.add_parser("train", help="Train world models and RL agents")
    train_parser.add_argument("--tickers", nargs="+", help="Specific tickers to train")
    train_parser.add_argument("--epochs", type=int, default=100, help="World model training epochs")
    train_parser.add_argument("--timesteps", type=int, default=1_000_000, help="RL training timesteps")
    train_parser.add_argument("--skip-rl", action="store_true", help="Skip RL agent training")

    # backtest
    bt_parser = subparsers.add_parser("backtest", help="Run backtests")
    bt_parser.add_argument("--tickers", nargs="+", help="Specific tickers to backtest")
    bt_parser.add_argument("--forensics", action="store_true", help="Run loss forensics after backtest")
    bt_parser.add_argument("--adapt", action="store_true", help="Auto-adapt parameters after forensics")

    # live
    subparsers.add_parser("live", help="Start live trading mode")

    # dashboard
    subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")

    # export
    export_parser = subparsers.add_parser("export", help="Export data to Excel")
    export_parser.add_argument("--output", help="Output directory")

    # status
    subparsers.add_parser("status", help="Show system status")

    args = parser.parse_args()

    # Setup logging
    setup_logger()

    if args.command is None:
        parser.print_help()
        return

    # Route to command
    commands = {
        "fetch": cmd_fetch,
        "train": cmd_train,
        "backtest": cmd_backtest,
        "live": cmd_live,
        "dashboard": cmd_dashboard,
        "export": cmd_export,
        "status": cmd_status,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
