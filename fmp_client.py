import os
from typing import Dict, List

import httpx
from dotenv import load_dotenv

load_dotenv()

FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_BASE_URL = os.getenv("FMP_BASE_URL", "https://financialmodelingprep.com/stable").rstrip("/")


def _require_api_key() -> str:
    if not FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY is not set. Please configure it in your environment or .env file.")
    return FMP_API_KEY


def _get_client() -> httpx.Client:
    return httpx.Client(base_url=FMP_BASE_URL, timeout=15.0)


def search_symbols(query: str) -> List[Dict]:
    """
    Call FMP Stock Symbol Search API:
    GET /stable/search-symbol?query={query}&apikey=FMP_API_KEY
    """
    if not query or not query.strip():
        return []

    _require_api_key()
    cleaned = query.strip()
    with _get_client() as client:
        resp = client.get("/search-symbol", params={"query": cleaned, "apikey": FMP_API_KEY})
        resp.raise_for_status()
        data = resp.json() or []

    results: List[Dict] = []
    for item in data:
        results.append(
            {
                "symbol": item.get("symbol") or item.get("ticker"),
                "name": item.get("name"),
                "exchange": item.get("exchangeShortName") or item.get("exchange"),
                "currency": item.get("currency") or item.get("currencyCode"),
            }
        )
    return [r for r in results if r.get("symbol")]


def get_transcript_dates(symbol: str) -> List[Dict]:
    """
    Call FMP Transcripts Dates By Symbol API:
    GET /stable/earning-call-transcript-dates?symbol={symbol}&apikey=FMP_API_KEY
    """
    if not symbol:
        return []

    _require_api_key()
    with _get_client() as client:
        resp = client.get("/earning-call-transcript-dates", params={"symbol": symbol, "apikey": FMP_API_KEY})
        resp.raise_for_status()
        data = resp.json() or []

    normalized: List[Dict] = []
    for item in data:
        year = item.get("fiscalYear") or item.get("year")
        quarter = item.get("fiscalQuarter") or item.get("quarter")
        date = item.get("date") or item.get("reportDate")
        normalized.append({"year": year, "quarter": quarter, "date": date})
    return normalized


def get_transcript(symbol: str, year: int, quarter: int) -> Dict:
    """
    Call FMP Earnings Transcript API.
    """
    _require_api_key()
    with _get_client() as client:
        resp = client.get(
            "/earning-call-transcript",
            params={"symbol": symbol, "year": year, "quarter": quarter, "apikey": FMP_API_KEY},
        )
        resp.raise_for_status()
        data = resp.json() or []

    if not data:
        raise ValueError(f"No transcript found for {symbol} FY{year} Q{quarter}")

    first = data[0]
    return {
        "symbol": symbol,
        "year": year,
        "quarter": quarter,
        "date": first.get("date") or first.get("reportDate"),
        "content": first.get("content") or "",
    }


def get_quarterly_financials(symbol: str, limit: int = 4) -> Dict:
    """
    Fetch recent quarterly financial statements.
    """
    _require_api_key()
    params = {"symbol": symbol, "period": "quarter", "limit": limit, "apikey": FMP_API_KEY}
    with _get_client() as client:
        income = client.get("/income-statement", params=params)
        balance = client.get("/balance-sheet-statement", params=params)
        cash_flow = client.get("/cash-flow-statement", params=params)

        income.raise_for_status()
        balance.raise_for_status()
        cash_flow.raise_for_status()

    return {
        "income": income.json() or [],
        "balance": balance.json() or [],
        "cashFlow": cash_flow.json() or [],
    }


def get_earnings_context(symbol: str, year: int, quarter: int) -> Dict:
    """
    Aggregate transcript and financials into a single context used by the analysis engine.
    """
    transcript = get_transcript(symbol, year, quarter)
    financials = get_quarterly_financials(symbol, limit=4)

    # TODO: Incorporate historical price context via FMP price APIs.
    price_window = []
    post_earnings_return = None

    return {
        "symbol": symbol,
        "year": year,
        "quarter": quarter,
        "transcript_text": transcript.get("content", ""),
        "transcript_date": transcript.get("date"),
        "financials": financials,
        "price_window": price_window,
        "post_earnings_return": post_earnings_return,
    }
