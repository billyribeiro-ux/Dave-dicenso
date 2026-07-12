"""
backend/api/routes/tickers.py — Ticker data endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from data.storage import DataStorage
from data.fetcher import FMPDataFetcher
from data.validation import DataValidator
from utils.config import get_config
from utils.logger import get_logger

router = APIRouter()
config = get_config()
logger = get_logger()
storage = DataStorage(config.DATA_DIR)
fetcher = FMPDataFetcher(config.get_api_key())
validator = DataValidator()


@router.get("/")
async def list_tickers():
    return {"tickers": config.TICKERS}


@router.get("/{ticker}/price")
async def get_current_price(ticker: str):
    if ticker not in config.TICKERS:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")
    try:
        quote = fetcher.fetch_real_time_quote(ticker)
        is_valid, issues = validator.validate_realtime_quote(quote)
        if not is_valid:
            logger.warning(f"Quote validation issues for {ticker}: {issues}")
        return {
            "ticker": ticker,
            "price": quote.get("price"),
            "change": quote.get("change"),
            "change_percent": quote.get("changePercentage"),
            "timestamp": quote.get("timestamp"),
            "is_valid": is_valid,
        }
    except Exception as e:
        logger.error(f"Error fetching price for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticker}/historical")
async def get_historical_prices(
    ticker: str,
    start_date: str = Query(..., description="YYYY-MM-DD"),
    end_date: str = Query(..., description="YYYY-MM-DD"),
):
    if ticker not in config.TICKERS:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")
    try:
        df = storage.load_tick_data(ticker, start_date, end_date)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {ticker} in range {start_date} to {end_date}")
        return {"ticker": ticker, "count": len(df), "data": df.to_dict(orient="records")}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading historical data for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticker}/range")
async def get_data_range(ticker: str):
    if ticker not in config.TICKERS:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")
    try:
        start, end = storage.get_available_date_range(ticker)
        return {"ticker": ticker, "start_date": str(start), "end_date": str(end)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prices/batch")
async def get_batch_prices():
    try:
        quotes = fetcher.fetch_batch_quotes(config.TICKERS)
        results = {}
        for ticker in config.TICKERS:
            if ticker in quotes:
                q = quotes[ticker]
                results[ticker] = {
                    "price": q.get("price"),
                    "change_percent": q.get("changePercentage"),
                    "timestamp": q.get("timestamp"),
                }
        return {"quotes": results, "timestamp": __import__('datetime').datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error fetching batch prices: {e}")
        raise HTTPException(status_code=500, detail=str(e))
