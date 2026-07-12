"""
backend/api/routes/system.py — System status and control endpoints.
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime
import time

from live.position_manager import PositionManager
from data.storage import DataStorage
from utils.config import get_config
from utils.logger import get_logger

router = APIRouter()
config = get_config()
logger = get_logger()
storage = DataStorage(config.DATA_DIR)
_start_time = time.time()


@router.get("/status")
async def get_system_status():
    try:
        pm = PositionManager()
        positions = pm.get_all_positions()
        trained = 0
        versions = {}
        for ticker in config.TICKERS:
            v = storage.get_latest_model_version(ticker)
            versions[ticker] = v
            if v != "0.0.0":
                trained += 1
        return {
            "status": "LIVE" if trained > 0 else "READY",
            "uptime_seconds": round(time.time() - _start_time, 1),
            "active_tickers": len(config.TICKERS),
            "trained_models": trained,
            "current_model_versions": versions,
            "last_backtest": None,
            "last_training": None,
            "active_positions": len(positions),
            "api_calls_today": 0,
            "errors_today": 0,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start-training")
async def start_training():
    try:
        return {"status": "Training initiated", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
