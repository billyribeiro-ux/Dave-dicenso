"""
utils/config.py — Singleton configuration using pydantic BaseSettings.
Loads FMP_API_KEY from environment variables via python-dotenv.
"""

import os
from datetime import date
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings

load_dotenv()


class Config(BaseSettings):
    """Singleton configuration for the trading machine.

    All attributes have sensible defaults. FMP_API_KEY is loaded from
    the environment variable of the same name.
    """

    TICKERS: List[str] = ["TSLA", "AMZN", "NVDA", "CSCO", "SPY", "QQQ", "IWM", "SPX", "NFLX"]
    START_DATE: str = "2018-01-01"
    END_DATE: str = "2026-07-10"
    DATA_DIR: str = "./data/storage"
    MODEL_DIR: str = "./models/saved"
    LOG_DIR: str = "./logs"
    DASHBOARD_PORT: int = 8501
    DASHBOARD_HOST: str = "localhost"
    MARKET_OPEN: str = "09:30"
    MARKET_CLOSE: str = "16:00"
    TIMEZONE: str = "US/Eastern"
    POLLING_INTERVAL_SECONDS: int = 1
    LATENT_DIM: int = 256
    INPUT_WINDOW_TICKS: int = 500
    FUTURE_WINDOW_TICKS: int = 100
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 0.0001
    PPO_EPOCHS: int = 10
    PPO_STEPS_PER_EPOCH: int = 2048
    INITIAL_CAPITAL: float = 100000.0
    MAX_DRAWDOWN_PCT: float = 0.50
    RISK_PER_TRADE_MIN: float = 0.0025
    RISK_PER_TRADE_MAX: float = 0.02
    DATABASE_URL: str = "sqlite:///./data/trading_machine.db"
    BACKTEST_DB_URL: str = "sqlite:///./data/backtest_results.db"
    LOSS_FORENSICS_DB_URL: str = "sqlite:///./data/loss_forensics.db"

    FMP_API_KEY: str = ""

    @field_validator("TICKERS")
    @classmethod
    def validate_tickers_not_empty(cls, v: List[str]) -> List[str]:
        if not v or len(v) == 0:
            raise ValueError("TICKERS list must not be empty")
        return v

    @field_validator("END_DATE")
    @classmethod
    def validate_dates(cls, v: str, info: Any) -> str:
        start = info.data.get("START_DATE", "2018-01-01")
        try:
            start_dt = date.fromisoformat(start)
            end_dt = date.fromisoformat(v)
            if end_dt <= start_dt:
                raise ValueError(f"END_DATE ({v}) must be after START_DATE ({start})")
        except (ValueError, TypeError):
            pass
        return v

    @field_validator("FMP_API_KEY")
    @classmethod
    def load_api_key(cls, v: str) -> str:
        if not v:
            v = os.environ.get("FMP_API_KEY", "")
        return v

    def get_api_key(self) -> str:
        """Return the FMP API key, loading from env if not already set."""
        if not self.FMP_API_KEY:
            self.FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
        return self.FMP_API_KEY

    def to_dict(self) -> Dict[str, Any]:
        """Return all configuration values as a flat dictionary."""
        return self.model_dump()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


_config_instance: Config | None = None


def get_config() -> Config:
    """Return the singleton Config instance, creating it if necessary."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def reset_config() -> Config:
    """Reset the singleton (useful for testing)."""
    global _config_instance
    _config_instance = Config()
    return _config_instance
