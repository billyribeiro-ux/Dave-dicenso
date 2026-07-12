"""
backend/api/routes/backtest.py — Backtest results and trade history endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import Optional
import io
import pandas as pd

from data.storage import DataStorage
from evolution.backtester import Backtester
from backend.api.models import ExportRequest
from utils.config import get_config
from utils.logger import get_logger

router = APIRouter()
config = get_config()
logger = get_logger()
storage = DataStorage(config.DATA_DIR)
backtester = Backtester(config)


@router.get("/summary")
async def get_backtest_summary(
    start_date: str = Query(..., description="YYYY-MM-DD"),
    end_date: str = Query(..., description="YYYY-MM-DD"),
    ticker: Optional[str] = Query(None),
):
    try:
        summary = backtester.get_summary(start_date, end_date, ticker)
        return summary
    except Exception as e:
        logger.error(f"Error getting backtest summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results")
async def get_backtest_results(
    start_date: str = Query(..., description="YYYY-MM-DD"),
    end_date: str = Query(..., description="YYYY-MM-DD"),
    ticker: Optional[str] = Query(None),
):
    try:
        results = backtester.get_full_results(start_date, end_date, ticker)
        return results
    except Exception as e:
        logger.error(f"Error getting backtest results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades")
async def get_backtest_trades(
    start_date: str = Query(..., description="YYYY-MM-DD"),
    end_date: str = Query(..., description="YYYY-MM-DD"),
    ticker: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
):
    try:
        trades_df = storage.load_trades(start_date, end_date, ticker)
        total = len(trades_df)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_trades = trades_df.iloc[start_idx:end_idx].to_dict(orient="records")
        return {"trades": page_trades, "total": total, "page": page, "page_size": page_size}
    except Exception as e:
        logger.error(f"Error loading trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export")
async def export_backtest(request: ExportRequest):
    try:
        trades_df = storage.load_trades(request.start_date, request.end_date, request.ticker)
        if request.format == "csv":
            output = io.StringIO()
            trades_df.to_csv(output, index=False)
            output.seek(0)
            filename = f"backtest_{request.start_date}_{request.end_date}.csv"
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        else:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                trades_df.to_excel(writer, sheet_name="Trades", index=False)
                try:
                    summary = backtester.get_summary(request.start_date, request.end_date, request.ticker)
                    pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
                except Exception:
                    pass
            output.seek(0)
            filename = f"backtest_{request.start_date}_{request.end_date}.xlsx"
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
    except Exception as e:
        logger.error(f"Error exporting backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/equity-curve")
async def get_equity_curve(
    start_date: str = Query(...),
    end_date: str = Query(...),
    ticker: Optional[str] = Query(None),
):
    try:
        curve = backtester.get_equity_curve(start_date, end_date, ticker)
        return {"equity_curve": curve}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ticker-breakdown")
async def get_ticker_breakdown(
    start_date: str = Query(...),
    end_date: str = Query(...),
):
    try:
        breakdown = backtester.get_ticker_breakdown(start_date, end_date)
        return {"breakdown": breakdown}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
