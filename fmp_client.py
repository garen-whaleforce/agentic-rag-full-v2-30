import os
from typing import Dict, List
from datetime import datetime, timedelta

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


def _get(client: httpx.Client, path: str, params: dict) -> httpx.Response:
    """
    Ensure we keep the /stable prefix; paths starting with "/" would drop it.
    """
    clean_path = path.lstrip("/")
    return client.get(clean_path, params=params)


def get_company_profile(symbol: str) -> Dict:
    """
    Fetch basic company profile (name, exchange, sector) for enrichment.
    """
    if not symbol:
        return {}
    _require_api_key()
    with _get_client() as client:
        # FMP stable profile expects symbol as query param, not path segment
        resp = _get(client, "profile", params={"symbol": symbol, "apikey": FMP_API_KEY})
        resp.raise_for_status()
        data = resp.json() or []
    if not data:
        raise ValueError(f"Company profile not found for {symbol}")

    first = data[0]
    return {
        "company": first.get("companyName") or first.get("name"),
        "exchange": first.get("exchangeShortName") or first.get("exchange"),
        "sector": first.get("sector"),
        "industry": first.get("industry"),
    }


def _historical_prices(symbol: str, start: datetime, end: datetime) -> List[dict]:
    """
    Fetch daily historical prices between start and end (inclusive).
    """
    _require_api_key()
    with _get_client() as client:
        resp = _get(
            client,
            "historical-price-full",
            params={
                "symbol": symbol,
                "from": start.strftime("%Y-%m-%d"),
                "to": end.strftime("%Y-%m-%d"),
                "apikey": FMP_API_KEY,
            },
        )
        resp.raise_for_status()
        data = resp.json() or {}
    hist = data.get("historical") or []
    # FMP returns descending by date; ensure sorted ascending
    hist_sorted = sorted(hist, key=lambda x: x.get("date", ""))
    return hist_sorted


def compute_post_return(symbol: str, call_date: str, days: int = 3) -> Dict[str, float | None]:
    """
    Compute post-earnings return using daily close prices:
      - start price = first trading day after call_date
      - end price   = trading day N after start
      - return = (end - start) / start

    Returns dict with start_date, end_date, start_price, end_price, return.
    """
    try:
        call_dt = datetime.fromisoformat(call_date)
    except Exception:
        try:
            call_dt = datetime.strptime(call_date, "%Y-%m-%d")
        except Exception:
            return {"return": None}

    start = call_dt + timedelta(days=0)
    end = call_dt + timedelta(days=days + 2)  # buffer to catch weekends/holidays
    prices = _historical_prices(symbol, start, end)
    if not prices:
        return {"return": None}

    # find first trading day after call_date
    start_row = next((p for p in prices if p.get("date") > call_dt.strftime("%Y-%m-%d")), None)
    if not start_row:
        return {"return": None}
    start_idx = prices.index(start_row)
    end_idx = start_idx + days
    if end_idx >= len(prices):
        return {"return": None}
    end_row = prices[end_idx]
    try:
        start_price = float(start_row.get("close"))
        end_price = float(end_row.get("close"))
    except Exception:
        return {"return": None}
    ret = (end_price - start_price) / start_price if start_price else None
    return {
        "start_date": start_row.get("date"),
        "end_date": end_row.get("date"),
        "start_price": start_price,
        "end_price": end_price,
        "return": ret,
    }


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
        resp = _get(client, "search-symbol", params={"query": cleaned, "apikey": FMP_API_KEY})
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
        resp = _get(client, "earning-call-transcript-dates", params={"symbol": symbol, "apikey": FMP_API_KEY})
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
        resp = _get(
            client,
            "earning-call-transcript",
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
        income = _get(client, "income-statement", params=params)
        balance = _get(client, "balance-sheet-statement", params=params)
        cash_flow = _get(client, "cash-flow-statement", params=params)

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
    profile = get_company_profile(symbol)
    transcript = get_transcript(symbol, year, quarter)
    financials = get_quarterly_financials(symbol, limit=4)

    # TODO: Incorporate historical price context via FMP price APIs.
    price_window = []
    post_earnings = compute_post_return(symbol, transcript.get("date") or "", days=3)
    post_earnings_return = post_earnings.get("return")

    return {
        "symbol": symbol,
        "year": year,
        "quarter": quarter,
        "company": profile.get("company"),
        "sector": profile.get("sector"),
        "exchange": profile.get("exchange"),
        "transcript_text": transcript.get("content", ""),
        "transcript_date": transcript.get("date"),
        "financials": financials,
        "price_window": price_window,
        "post_earnings_return": post_earnings_return,
        "post_return_meta": post_earnings,
    }
