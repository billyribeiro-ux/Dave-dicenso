"""
dashboard/exports.py — Export backtest results, trades, and reports to Excel.

Uses openpyxl for Excel workbook creation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from data.storage import DataStorage
from utils.config import get_config
from utils.logger import get_logger


class Exports:
    """Export trading data to Excel format for external analysis."""

    def __init__(self, storage: Optional[DataStorage] = None):
        self.config = get_config()
        self.storage = storage or DataStorage(self.config.DATA_DIR)
        self.logger = get_logger()

    def export_trades_to_excel(
        self, filepath: str, ticker: Optional[str] = None
    ) -> str:
        """Export trade history to an Excel file.

        Args:
            filepath: Path for the output .xlsx file.
            ticker: Optional ticker filter.

        Returns:
            Absolute path to the created file.
        """
        trades = self.storage.load_trades(ticker=ticker)

        if len(trades) == 0:
            self.logger.warning("No trades to export")
            return ""

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            # Trades sheet
            trades.to_excel(writer, sheet_name="Trades", index=False)

            # Summary sheet
            if len(trades) > 0:
                summary = pd.DataFrame([{
                    "Total Trades": len(trades),
                    "Winning Trades": int((trades["pnl"] > 0).sum()),
                    "Losing Trades": int((trades["pnl"] < 0).sum()),
                    "Win Rate": round((trades["pnl"] > 0).mean() * 100, 1),
                    "Total P&L": round(trades["pnl"].sum(), 2),
                    "Avg Win": round(trades[trades["pnl"] > 0]["pnl"].mean(), 2),
                    "Avg Loss": round(trades[trades["pnl"] < 0]["pnl"].mean(), 2),
                    "Largest Win": round(trades["pnl"].max(), 2),
                    "Largest Loss": round(trades["pnl"].min(), 2),
                }])
                summary.to_excel(writer, sheet_name="Summary", index=False)

            # By ticker sheet
            if "ticker" in trades.columns and len(trades["ticker"].unique()) > 1:
                by_ticker = trades.groupby("ticker").agg(
                    total_trades=("pnl", "count"),
                    win_rate=("pnl", lambda x: (x > 0).mean() * 100),
                    total_pnl=("pnl", "sum"),
                    avg_pnl=("pnl", "mean"),
                ).reset_index()
                by_ticker.to_excel(writer, sheet_name="By Ticker", index=False)

        self.logger.info(f"Exported trades to {filepath}")
        return str(Path(filepath).absolute())

    def export_backtest_to_excel(self, filepath: str) -> str:
        """Export backtest results to Excel."""
        results = self.storage.load_backtest_results(limit=100)

        if len(results) == 0:
            self.logger.warning("No backtest results to export")
            return ""

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            results.to_excel(writer, sheet_name="Backtests", index=False)

            # Summary
            if len(results) > 0:
                summary = pd.DataFrame([{
                    "Total Runs": len(results),
                    "Best Win Rate": round(results["win_rate"].max() * 100, 1),
                    "Best Sharpe": round(results["sharpe_ratio"].max(), 2),
                    "Best Profit Factor": round(results["profit_factor"].max(), 2),
                    "Avg Net Profit": round(results["net_profit"].mean(), 2),
                    "Min Drawdown": round(results["max_drawdown_pct"].min() * 100, 1),
                }])
                summary.to_excel(writer, sheet_name="Summary", index=False)

        self.logger.info(f"Exported backtest results to {filepath}")
        return str(Path(filepath).absolute())

    def export_forensics_to_excel(self, filepath: str) -> str:
        """Export forensics reports to Excel."""
        forensics = self.storage.load_loss_forensics(limit=50)

        if len(forensics) == 0:
            self.logger.warning("No forensics reports to export")
            return ""

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            # Parse JSON columns
            for col in ["classification_counts", "recommendations"]:
                if col in forensics.columns:
                    forensics[col] = forensics[col].apply(
                        lambda x: json.dumps(json.loads(x) if isinstance(x, str) else x, indent=2)
                        if x else ""
                    )

            forensics.to_excel(writer, sheet_name="Forensics", index=False)

        self.logger.info(f"Exported forensics to {filepath}")
        return str(Path(filepath).absolute())

    def export_all(self, output_dir: str) -> dict:
        """Export all data types to an output directory.

        Returns dict mapping data type to filepath.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        exports = {}

        trades_path = output_dir / f"trades_{timestamp}.xlsx"
        result = self.export_trades_to_excel(str(trades_path))
        if result:
            exports["trades"] = result

        bt_path = output_dir / f"backtests_{timestamp}.xlsx"
        result = self.export_backtest_to_excel(str(bt_path))
        if result:
            exports["backtests"] = result

        forensics_path = output_dir / f"forensics_{timestamp}.xlsx"
        result = self.export_forensics_to_excel(str(forensics_path))
        if result:
            exports["forensics"] = result

        return exports
