"""
backend/api/models.py — Pydantic models for API request/response validation.
"""

from datetime import datetime
from typing import Optional, Literal, Dict, Any, List

from pydantic import BaseModel, Field


class TradeRecord(BaseModel):
    id: int
    ticker: str
    direction: Literal["LONG", "SHORT"]
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    loss_classification: Optional[str] = None
    model_version: str
    created_at: datetime

    class Config:
        from_attributes = True


class TradeListResponse(BaseModel):
    trades: list[TradeRecord]
    total: int
    page: int
    page_size: int


class SignalResponse(BaseModel):
    ticker: str
    signal: Literal["BUY", "SELL", "NEUTRAL"]
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    confidence: float
    timestamp: datetime
    regime: Optional[str] = None
    latent_state_version: str


class PositionResponse(BaseModel):
    ticker: str
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    entry_time: datetime
    duration_minutes: float


class BacktestSummaryResponse(BaseModel):
    ticker: Optional[str] = None
    start_date: str
    end_date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    loss_rate: float
    total_wins: float
    total_losses: float
    net_profit: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_loss_ratio: float
    model_version: str


class BacktestResultResponse(BaseModel):
    summary: BacktestSummaryResponse
    equity_curve: list[dict]
    drawdown_curve: list[dict]
    ticker_breakdown: list[dict]


class LearningLogEntry(BaseModel):
    id: int
    ticker: str
    date: str
    classification: str
    description: str
    action_taken: str
    result: Optional[str] = None
    created_at: datetime


class SystemStatusResponse(BaseModel):
    status: Literal["READY", "TRAINING", "BACKTESTING", "LIVE", "ERROR"]
    uptime_seconds: float
    active_tickers: int
    trained_models: int
    current_model_versions: dict[str, str]
    last_backtest: Optional[datetime] = None
    last_training: Optional[datetime] = None
    active_positions: int
    api_calls_today: int
    errors_today: int


class DateRangeRequest(BaseModel):
    start_date: str
    end_date: str
    ticker: Optional[str] = None


class ExportRequest(BaseModel):
    start_date: str
    end_date: str
    ticker: Optional[str] = None
    format: Literal["csv", "excel"]
