"""
backend/api/routes/learning.py — Learning log and forensics endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import sqlite3
import pandas as pd

from evolution.regime_detector import RegimeDetector
from data.storage import DataStorage
from utils.config import get_config
from utils.logger import get_logger

router = APIRouter()
config = get_config()
logger = get_logger()
storage = DataStorage(config.DATA_DIR)


@router.get("/log")
async def get_learning_log(
    ticker: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
):
    try:
        db_path = config.LOSS_FORENSICS_DB_URL.replace("sqlite:///", "")
        conn = sqlite3.connect(db_path)
        query = "SELECT * FROM loss_forensics"
        params = []
        if ticker:
            query += " WHERE ticker = ?"
            params.append(ticker)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return {"entries": df.to_dict(orient="records"), "count": len(df)}
    except Exception as e:
        logger.error(f"Error loading learning log: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/regimes")
async def get_regimes(ticker: Optional[str] = Query(None)):
    try:
        detector = RegimeDetector()
        regimes = detector.get_current_regimes()
        if ticker:
            regimes = {ticker: regimes.get(ticker)} if ticker in regimes else {}
        return {"regimes": regimes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-versions")
async def get_model_versions():
    try:
        versions = {}
        for ticker in config.TICKERS:
            versions[ticker] = storage.get_latest_model_version(ticker)
        return {"versions": versions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
