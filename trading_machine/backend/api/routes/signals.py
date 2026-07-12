"""
backend/api/routes/signals.py — Trading signals and positions endpoints.
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime

from models.ticker_manager import TickerManager
from live.screener import Screener
from live.position_manager import PositionManager
from api.websocket import ws_manager
from utils.config import get_config
from utils.logger import get_logger

router = APIRouter()
config = get_config()
logger = get_logger()
ticker_manager = TickerManager(config)
ticker_manager.load_all_models()
screener = Screener(ticker_manager, config)
position_manager = PositionManager()


@router.get("/live")
async def get_live_signals():
    try:
        signals_list = screener.get_all_signals()
        return {"signals": signals_list, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error getting live signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/live/{ticker}")
async def get_ticker_signal(ticker: str):
    if ticker not in config.TICKERS:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")
    try:
        signal = screener.get_signal(ticker)
        if signal is None:
            return {"ticker": ticker, "signal": "NEUTRAL", "confidence": 0.0, "timestamp": datetime.now().isoformat()}
        return signal
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions")
async def get_positions():
    try:
        positions = position_manager.get_all_positions()
        return {"positions": positions, "count": len(positions), "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions/{ticker}")
async def get_ticker_position(ticker: str):
    if ticker not in config.TICKERS:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")
    try:
        position = position_manager.get_position(ticker)
        if position is None:
            return {"ticker": ticker, "has_position": False}
        return {"ticker": ticker, "has_position": True, "position": position}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
