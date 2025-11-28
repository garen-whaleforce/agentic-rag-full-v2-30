import asyncio
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Literal, Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from uuid import uuid4

from analysis_engine import analyze_earnings, analyze_earnings_async
from fmp_client import (
    close_fmp_client,
    close_fmp_async_client,
    get_earnings_calendar_for_date,
    get_earnings_calendar_for_range,
    get_transcript_dates,
    search_symbols,
    _require_api_key,
)
from agentic_rag_bridge import verify_agentic_repo
from storage import ensure_db_writable, get_call, init_db, list_calls

load_dotenv()

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Route B: Real-time Earnings Call Analysis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def _shutdown_clients():
    # Ensure shared HTTP client is closed on app shutdown
    close_fmp_client()
    await close_fmp_async_client()


@app.on_event("startup")
def _startup_checks():
    # Fail fast if required config is missing
    _require_api_key()
    # Ensure DB path exists/writable
    ensure_db_writable()
    # Ensure external repo/credentials exist
    verify_agentic_repo()


@app.on_event("startup")
async def _schedule_earnings_calendar_prefetch():
    tz = None
    try:
        tz = ZoneInfo("America/New_York")
    except Exception:
        pass

    async def prefetch_loop():
        while True:
            try:
                if tz is not None:
                    now = datetime.now(tz)
                else:
                    now = datetime.utcnow()
                target = now.replace(hour=6, minute=0, second=0, microsecond=0)
                if target <= now:
                    target = target + timedelta(days=1)
                wait_seconds = (target - now).total_seconds()
                if wait_seconds < 0:
                    wait_seconds = 60.0
                await asyncio.sleep(wait_seconds)

                if tz is not None:
                    today = target.date()
                else:
                    today = datetime.utcnow().date()
                start = today - timedelta(days=7)
                try:
                    get_earnings_calendar_for_range(
                        start_date=start.isoformat(),
                        end_date=today.isoformat(),
                        min_market_cap=10_000_000_000,
                        skip_cache=True,
                    )
                except Exception as exc:
                    logging.exception("Failed to prefetch earnings calendar range: %s", exc)
            except Exception as exc:
                logging.exception("Unexpected error in prefetch_loop: %s", exc)
                await asyncio.sleep(60.0)

    asyncio.create_task(prefetch_loop())


class AnalyzeRequest(BaseModel):
    symbol: str = Field(..., description="Ticker symbol, e.g., AAPL")
    year: int = Field(..., description="Fiscal year")
    quarter: int = Field(..., description="Fiscal quarter (1-4)")
    main_model: Optional[Literal["gpt-5.1", "gpt-5-mini", "gpt-4o-mini"]] = Field(
        None, description="Main agent model override"
    )
    helper_model: Optional[Literal["gpt-5-mini", "gpt-4o-mini"]] = Field(
        None, description="Helper agents model override"
    )

class BatchAnalyzeRequest(BaseModel):
    tickers: list[str] = Field(..., description="List of ticker symbols")
    latest_only: bool = Field(True, description="Always pick latest available quarter")


@app.get("/api/symbols")
def api_symbols(q: str = Query("", description="Search term for company name or ticker")) -> List[dict]:
    try:
        return search_symbols(q)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/earnings-calendar/range")
def api_earnings_calendar_range(
    start_date: Optional[str] = Query(
        None,
        description="Start date YYYY-MM-DD; default: today-7 (US/Eastern)",
    ),
    end_date: Optional[str] = Query(
        None,
        description="End date YYYY-MM-DD; default: today (US/Eastern)",
    ),
    min_market_cap: float = Query(
        10_000_000_000,
        description="Minimum market cap filter",
    ),
    refresh: bool = Query(
        False,
        description="Skip cache and refetch underlying data",
    ),
):
    try:
        tz = ZoneInfo("America/New_York")
    except Exception:
        # fallback: UTC
        tz = None

    if tz is not None:
        now = datetime.now(tz).date()
    else:
        now = datetime.utcnow().date()
    default_end = now
    default_start = now - timedelta(days=7)

    if end_date:
        try:
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="end_date must be YYYY-MM-DD")
    else:
        end = default_end

    if start_date:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="start_date must be YYYY-MM-DD")
    else:
        start = default_start

    if start > end:
        start, end = end, start

    try:
        return get_earnings_calendar_for_range(
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            min_market_cap=min_market_cap,
            skip_cache=refresh,
        )
    except Exception as exc:  # noqa: BLE001
        logging.exception("Error in api_earnings_calendar_range: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/transcript-dates")
def api_transcript_dates(symbol: str = Query(..., description="Ticker symbol")) -> List[dict]:
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol is required")
    try:
        return get_transcript_dates(symbol)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/earnings-calendar/today")
def api_earnings_calendar_today(
    min_market_cap: float = Query(10_000_000_000, description="Minimum market cap filter"),
    date: Optional[str] = Query(None, description="Target date YYYY-MM-DD; defaults to today (UTC)"),
    refresh: bool = Query(False, description="Skip cache and refetch"),
):
    date_str = None
    if date:
        try:
            datetime.strptime(date, "%Y-%m-%d")
            date_str = date
        except ValueError:
            raise HTTPException(status_code=400, detail="date must be YYYY-MM-DD")
    try:
        return get_earnings_calendar_for_date(
            target_date=date_str, min_market_cap=min_market_cap, skip_cache=refresh
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/analyze")
async def api_analyze(payload: AnalyzeRequest):
    try:
        result = await analyze_earnings_async(
            payload.symbol,
            payload.year,
            payload.quarter,
            payload.main_model,
            payload.helper_model,
        )
        return JSONResponse(result)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/batch-analyze")
async def api_batch_analyze(payload: BatchAnalyzeRequest):
    """
    Simple batch endpoint: takes tickers, fetches latest quarter per ticker (if latest_only),
    and analyzes sequentially. Returns results inline for simplicity.
    """
    tickers = [t.strip().upper() for t in payload.tickers if t.strip()]
    if not tickers:
        raise HTTPException(status_code=400, detail="tickers is required")

    results: List[Dict] = []

    async def _analyze_one(sym: str):
        try:
            # latest quarter if requested
            year = None
            quarter = None
            if payload.latest_only:
                dates = get_transcript_dates(sym)
                valid = [d for d in dates if d.get("year") and d.get("quarter")]
                if not valid:
                    return {"symbol": sym, "status": "error", "error": "no transcript dates"}
                latest = sorted(valid, key=lambda x: (int(x["year"]), int(x["quarter"])), reverse=True)[0]
                year = int(latest["year"])
                quarter = int(latest["quarter"])
            res = await analyze_earnings_async(sym, year, quarter) if year and quarter else None
            return {
                "symbol": sym,
                "status": "ok",
                "payload": res,
            }
        except Exception as exc:  # noqa: BLE001
            return {"symbol": sym, "status": "error", "error": str(exc)}

    for sym in tickers:
        results.append(await _analyze_one(sym))

    return {"results": results}


@app.get("/api/calls")
def api_calls(
    symbol: Optional[str] = Query(None),
    sector: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    ret_min: Optional[float] = Query(None),
    ret_max: Optional[float] = Query(None),
    prediction: Optional[str] = Query(None),
    correct: Optional[bool] = Query(None),
    sort: str = Query("date_desc"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    try:
        rows = list_calls(
            symbol=symbol,
            sector=sector,
            date_from=date_from,
            date_to=date_to,
            ret_min=ret_min,
            ret_max=ret_max,
            prediction=prediction,
            correct=correct,
            sort=sort,
            limit=limit,
            offset=offset,
        )
        return rows
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/call/{job_id}")
def api_call_detail(job_id: str):
    try:
        row = get_call(job_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if not row:
        raise HTTPException(status_code=404, detail="call not found")
    return row


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )
