"""
data/storage.py — Data persistence layer using PyArrow Parquet + SQLite/SQLAlchemy.
Stores tick data, trades, backtest results, model metadata, and loss forensics.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from utils.logger import get_logger

Base = declarative_base()


# ---------------------------------------------------------------------------
# SQLAlchemy ORM models
# ---------------------------------------------------------------------------

class DataMetadata(Base):
    __tablename__ = "data_metadata"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), unique=True, nullable=False)
    last_updated = Column(DateTime, default=datetime.utcnow)
    row_count = Column(Integer, default=0)
    date_range_start = Column(String(20), default="")
    date_range_end = Column(String(20), default="")


class TradeRecord(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False)
    direction = Column(String(10), nullable=False)
    entry_time = Column(DateTime, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_time = Column(DateTime, nullable=True)
    exit_price = Column(Float, nullable=True)
    pnl = Column(Float, default=0.0)
    pnl_pct = Column(Float, default=0.0)
    exit_reason = Column(String(50), default="")
    loss_classification = Column(String(20), default="")
    model_version = Column(String(20), default="")
    created_at = Column(DateTime, default=datetime.utcnow)


class BacktestResult(Base):
    __tablename__ = "backtest_results"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False)
    run_date = Column(DateTime, default=datetime.utcnow)
    win_rate = Column(Float, default=0.0)
    loss_rate = Column(Float, default=0.0)
    total_trades = Column(Integer, default=0)
    total_wins = Column(Integer, default=0)
    total_losses = Column(Integer, default=0)
    total_win_amount = Column(Float, default=0.0)
    total_loss_amount = Column(Float, default=0.0)
    net_profit = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    avg_win = Column(Float, default=0.0)
    avg_loss = Column(Float, default=0.0)
    largest_win = Column(Float, default=0.0)
    largest_loss = Column(Float, default=0.0)
    max_drawdown_pct = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    model_version = Column(String(20), default="")


class ModelMetadata(Base):
    __tablename__ = "model_metadata"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False)
    version = Column(String(20), nullable=False)
    filepath = Column(String(500), default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    metrics = Column(Text, default="{}")


class LossForensicsRecord(Base):
    __tablename__ = "loss_forensics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False)
    date = Column(DateTime, default=datetime.utcnow)
    classification_counts = Column(Text, default="{}")
    recommendations = Column(Text, default="{}")
    latent_clusters = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class DataNotFoundError(Exception):
    """Raised when requested data does not exist in storage."""
    pass


# ---------------------------------------------------------------------------
# DataStorage class
# ---------------------------------------------------------------------------

class DataStorage:
    """Data persistence manager for tick data, trades, and metadata.

    - Tick data: PyArrow Parquet format, one file per ticker.
    - Metadata + trades + results: SQLite via SQLAlchemy.
    """

    def __init__(self, base_dir: str):
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._logger = get_logger()

        # Determine database paths relative to base_dir
        db_dir = self._base_dir.parent
        db_dir.mkdir(parents=True, exist_ok=True)

        # Main database
        db_path = db_dir / "trading_machine.db"
        self._engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self._engine)
        self._SessionLocal = sessionmaker(bind=self._engine)

        # Backtest database
        bt_path = db_dir / "backtest_results.db"
        self._bt_engine = create_engine(f"sqlite:///{bt_path}", echo=False)
        Base.metadata.create_all(self._bt_engine)
        self._BTSessionLocal = sessionmaker(bind=self._bt_engine)

        # Loss forensics database
        lf_path = db_dir / "loss_forensics.db"
        self._lf_engine = create_engine(f"sqlite:///{lf_path}", echo=False)
        Base.metadata.create_all(self._lf_engine)
        self._LFSessionLocal = sessionmaker(bind=self._lf_engine)

    # ------------------------------------------------------------------
    # Tick data (Parquet)
    # ------------------------------------------------------------------

    def _ticker_dir(self, ticker: str) -> Path:
        d = self._base_dir / ticker
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _ticker_parquet_path(self, ticker: str) -> Path:
        return self._ticker_dir(ticker) / "tick_data.parquet"

    def save_tick_data(self, ticker: str, df: pd.DataFrame) -> bool:
        """Save a DataFrame to Parquet. Appends to existing data, removes duplicate timestamps.

        Also updates metadata in SQLite.
        """
        path = self._ticker_parquet_path(ticker)

        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=["timestamp"], keep="first")
            df = df.sort_values("timestamp").reset_index(drop=True)

        # Ensure timestamp column exists
        if "timestamp" not in df.columns:
            self._logger.error(f"No timestamp column in data for {ticker}")
            return False

        df.to_parquet(path, index=False)

        # Update metadata
        min_ts = df["timestamp"].min()
        max_ts = df["timestamp"].max()
        row_count = len(df)

        with Session(self._engine) as session:
            meta = session.query(DataMetadata).filter_by(ticker=ticker).first()
            if meta is None:
                meta = DataMetadata(ticker=ticker)
                session.add(meta)
            meta.last_updated = datetime.utcnow()
            meta.row_count = row_count
            meta.date_range_start = str(min_ts)
            meta.date_range_end = str(max_ts)
            session.commit()

        self._logger.info(f"Saved {row_count} rows for {ticker} to {path}")
        return True

    def load_tick_data(
        self, ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Load tick data from Parquet, optionally filtered by date range."""
        path = self._ticker_parquet_path(ticker)
        if not path.exists():
            self._logger.warning(f"No data file for {ticker}")
            return pd.DataFrame()

        df = pd.read_parquet(path)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            if start_date:
                df = df[df["timestamp"] >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df["timestamp"] <= pd.Timestamp(end_date)]

        return df.reset_index(drop=True)

    def get_available_date_range(self, ticker: str) -> Tuple[Optional[str], Optional[str]]:
        """Return (min_date, max_date) for stored ticker data."""
        with Session(self._engine) as session:
            meta = session.query(DataMetadata).filter_by(ticker=ticker).first()
            if meta and meta.date_range_start:
                return (str(meta.date_range_start), str(meta.date_range_end))

        # Fallback: scan Parquet
        path = self._ticker_parquet_path(ticker)
        if path.exists():
            df = pd.read_parquet(path)
            if "timestamp" in df.columns and len(df) > 0:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                return (str(df["timestamp"].min()), str(df["timestamp"].max()))

        return (None, None)

    def get_close_prices(
        self, ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> np.ndarray:
        """Load tick data and return ONLY the 'close' column as a numpy array.

        This is the ONLY data fed to the machine learning models.
        No open, high, low, or volume data returned.
        """
        df = self.load_tick_data(ticker, start_date, end_date)
        if len(df) == 0:
            raise DataNotFoundError(f"No data for {ticker}")
        if "close" not in df.columns:
            raise DataNotFoundError(f"No 'close' column in data for {ticker}")
        return df["close"].to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------
    # Trades (SQLite)
    # ------------------------------------------------------------------

    def save_trades(self, trades_df: pd.DataFrame) -> List[int]:
        """Save trade records to SQLite 'trades' table. Returns list of inserted IDs."""
        ids = []
        with Session(self._engine) as session:
            for _, row in trades_df.iterrows():
                trade = TradeRecord(
                    ticker=row.get("ticker", ""),
                    direction=row.get("direction", ""),
                    entry_time=row.get("entry_time"),
                    entry_price=row.get("entry_price", 0.0),
                    exit_time=row.get("exit_time"),
                    exit_price=row.get("exit_price", 0.0),
                    pnl=row.get("pnl", 0.0),
                    pnl_pct=row.get("pnl_pct", 0.0),
                    exit_reason=str(row.get("exit_reason", "")),
                    loss_classification=str(row.get("loss_classification", "")),
                    model_version=str(row.get("model_version", "")),
                )
                session.add(trade)
                session.flush()
                ids.append(trade.id)
            session.commit()
        return ids

    def load_trades(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """Query trades table with optional filters. Returns DataFrame."""
        with Session(self._engine) as session:
            query = session.query(TradeRecord)
            if ticker:
                query = query.filter(TradeRecord.ticker == ticker)
            if start_date:
                query = query.filter(TradeRecord.entry_time >= pd.Timestamp(start_date))
            if end_date:
                query = query.filter(TradeRecord.entry_time <= pd.Timestamp(end_date))
            trades = query.order_by(TradeRecord.entry_time.desc()).all()

        if not trades:
            return pd.DataFrame()

        records = []
        for t in trades:
            records.append({
                "id": t.id,
                "ticker": t.ticker,
                "direction": t.direction,
                "entry_time": t.entry_time,
                "entry_price": t.entry_price,
                "exit_time": t.exit_time,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "exit_reason": t.exit_reason,
                "loss_classification": t.loss_classification,
                "model_version": t.model_version,
                "created_at": t.created_at,
            })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Backtest results (SQLite)
    # ------------------------------------------------------------------

    def save_backtest_result(self, result_dict: dict) -> int:
        """Save backtest result to backtest_results.db. Returns inserted ID."""
        with Session(self._bt_engine) as session:
            bt = BacktestResult(
                ticker=result_dict.get("ticker", ""),
                run_date=result_dict.get("run_date", datetime.utcnow()),
                win_rate=result_dict.get("win_rate", 0.0),
                loss_rate=result_dict.get("loss_rate", 0.0),
                total_trades=result_dict.get("total_trades", 0),
                total_wins=result_dict.get("total_wins", 0),
                total_losses=result_dict.get("total_losses", 0),
                total_win_amount=result_dict.get("total_win_amount", 0.0),
                total_loss_amount=result_dict.get("total_loss_amount", 0.0),
                net_profit=result_dict.get("net_profit", 0.0),
                profit_factor=result_dict.get("profit_factor", 0.0),
                avg_win=result_dict.get("avg_win", 0.0),
                avg_loss=result_dict.get("avg_loss", 0.0),
                largest_win=result_dict.get("largest_win", 0.0),
                largest_loss=result_dict.get("largest_loss", 0.0),
                max_drawdown_pct=result_dict.get("max_drawdown_pct", 0.0),
                sharpe_ratio=result_dict.get("sharpe_ratio", 0.0),
                model_version=str(result_dict.get("model_version", "")),
            )
            session.add(bt)
            session.commit()
            return bt.id

    def load_backtest_results(
        self, ticker: Optional[str] = None, limit: int = 10
    ) -> pd.DataFrame:
        """Query backtest_results table, ordered by run_date descending."""
        with Session(self._bt_engine) as session:
            query = session.query(BacktestResult)
            if ticker:
                query = query.filter(BacktestResult.ticker == ticker)
            results = query.order_by(BacktestResult.run_date.desc()).limit(limit).all()

        if not results:
            return pd.DataFrame()

        records = []
        for r in results:
            records.append({
                "id": r.id,
                "ticker": r.ticker,
                "run_date": r.run_date,
                "win_rate": r.win_rate,
                "loss_rate": r.loss_rate,
                "total_trades": r.total_trades,
                "total_wins": r.total_wins,
                "total_losses": r.total_losses,
                "total_win_amount": r.total_win_amount,
                "total_loss_amount": r.total_loss_amount,
                "net_profit": r.net_profit,
                "profit_factor": r.profit_factor,
                "avg_win": r.avg_win,
                "avg_loss": r.avg_loss,
                "largest_win": r.largest_win,
                "largest_loss": r.largest_loss,
                "max_drawdown_pct": r.max_drawdown_pct,
                "sharpe_ratio": r.sharpe_ratio,
                "model_version": r.model_version,
            })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Model metadata (SQLite)
    # ------------------------------------------------------------------

    def save_model_metadata(
        self, ticker: str, version: str, filepath: str, metrics_dict: dict
    ) -> int:
        """Save model metadata. Returns inserted ID."""
        with Session(self._engine) as session:
            meta = ModelMetadata(
                ticker=ticker,
                version=version,
                filepath=filepath,
                metrics=json.dumps(metrics_dict),
            )
            session.add(meta)
            session.commit()
            return meta.id

    def get_latest_model_version(self, ticker: str) -> str:
        """Return the version string of the most recent model for the ticker.

        Returns "0.0.0" if no models exist.
        """
        with Session(self._engine) as session:
            meta = (
                session.query(ModelMetadata)
                .filter_by(ticker=ticker)
                .order_by(ModelMetadata.created_at.desc())
                .first()
            )
            if meta is None:
                return "0.0.0"
            return meta.version

    # ------------------------------------------------------------------
    # Loss forensics (SQLite)
    # ------------------------------------------------------------------

    def save_loss_forensics(self, ticker: str, report: dict) -> int:
        """Save forensics report to loss_forensics.db."""
        with Session(self._lf_engine) as session:
            rec = LossForensicsRecord(
                ticker=ticker,
                date=datetime.utcnow(),
                classification_counts=json.dumps(report.get("classification_counts", {})),
                recommendations=json.dumps(report.get("recommendations", [])),
                latent_clusters=json.dumps(report.get("latent_clusters", [])),
            )
            session.add(rec)
            session.commit()
            return rec.id

    def load_loss_forensics(
        self, ticker: Optional[str] = None, limit: int = 20
    ) -> pd.DataFrame:
        """Load forensics records."""
        with Session(self._lf_engine) as session:
            query = session.query(LossForensicsRecord)
            if ticker:
                query = query.filter(LossForensicsRecord.ticker == ticker)
            records = query.order_by(LossForensicsRecord.created_at.desc()).limit(limit).all()

        if not records:
            return pd.DataFrame()

        rows = []
        for r in records:
            rows.append({
                "id": r.id,
                "ticker": r.ticker,
                "date": r.date,
                "classification_counts": r.classification_counts,
                "recommendations": r.recommendations,
                "latent_clusters": r.latent_clusters,
                "created_at": r.created_at,
            })
        return pd.DataFrame(rows)
