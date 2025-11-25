import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from analysis_engine import analyze_earnings
from fmp_client import close_fmp_client, get_transcript_dates, search_symbols, _require_api_key
from storage import get_call, list_calls, init_db

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
def _shutdown_clients():
    # Ensure shared HTTP client is closed on app shutdown
    close_fmp_client()


@app.on_event("startup")
def _startup_checks():
    # Fail fast if required config is missing
    _require_api_key()


class AnalyzeRequest(BaseModel):
    symbol: str = Field(..., description="Ticker symbol, e.g., AAPL")
    year: int = Field(..., description="Fiscal year")
    quarter: int = Field(..., description="Fiscal quarter (1-4)")


@app.get("/api/symbols")
def api_symbols(q: str = Query("", description="Search term for company name or ticker")) -> List[dict]:
    try:
        return search_symbols(q)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/transcript-dates")
def api_transcript_dates(symbol: str = Query(..., description="Ticker symbol")) -> List[dict]:
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol is required")
    try:
        return get_transcript_dates(symbol)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/analyze")
def api_analyze(payload: AnalyzeRequest):
    try:
        result = analyze_earnings(payload.symbol, payload.year, payload.quarter)
        return JSONResponse(result)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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
