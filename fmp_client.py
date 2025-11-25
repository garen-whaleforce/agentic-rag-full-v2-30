import os
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import httpx
from dotenv import load_dotenv

load_dotenv()

FMP_API_KEY = os.getenv("FMP_API_KEY")
FMP_BASE_URL = os.getenv("FMP_BASE_URL", "https://financialmodelingprep.com/stable").rstrip("/")
_CLIENT: Optional[httpx.Client] = None


def _require_api_key() -> str:
    if not FMP_API_KEY:
        raise RuntimeError("FMP_API_KEY is not set. Please configure it in your environment or .env file.")
    return FMP_API_KEY


def _get_client() -> httpx.Client:
    """
    Lazy singleton httpx client so connections can be pooled across requests.
    """
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.Client(base_url=FMP_BASE_URL, timeout=10.0)
    return _CLIENT


def close_fmp_client() -> None:
    global _CLIENT
    if _CLIENT is not None:
        try:
            _CLIENT.close()
        finally:
            _CLIENT = None


def _get(client: httpx.Client, path: str, params: dict) -> httpx.Response:
    """
    Ensure we keep the /stable prefix; paths starting with "/" would drop it.
    """
    clean_path = path.lstrip("/")
    retry_status = {429, 500, 502, 503}
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = client.get(clean_path, params=params)
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            if exc.response.status_code in retry_status and attempt < 2:
                time.sleep(0.5 * (2**attempt))
                continue
            raise
        except httpx.RequestError as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(0.5 * (2**attempt))
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("Unexpected HTTP error without exception")


def get_company_profile(symbol: str) -> Dict:
    """
    Fetch basic company profile (name, exchange, sector) for enrichment.
    """
    if not symbol:
        return {}
    _require_api_key()
    client = _get_client()
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
    use_server_window = True
    params = {
        "symbol": symbol,
        "from": start.strftime("%Y-%m-%d"),
        "to": end.strftime("%Y-%m-%d"),
        "apikey": FMP_API_KEY,
    }
    client = _get_client()
    try:
        resp = _get(
            client,
            "historical-price-eod/full",
            params=params,
        )
        resp.raise_for_status()
        data = resp.json() or {}
    except httpx.HTTPStatusError:
        # If data not yet available or 404, return empty
        data = {}

    # fallback: if empty, try without date filters to get recent window then filter locally
    if (isinstance(data, dict) and not data.get("historical")) or (isinstance(data, list) and not data):
        try:
            use_server_window = False
            resp = _get(client, "historical-price-eod/full", params={"symbol": symbol, "apikey": FMP_API_KEY})
            resp.raise_for_status()
            data = resp.json() or {}
        except Exception:
            data = {}

    if isinstance(data, dict):
        hist = data.get("historical") or []
    elif isinstance(data, list):
        hist = data
    else:
        hist = []

    # If fetched without date filters, slice to requested window
    if hist and not use_server_window:
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
        hist = [h for h in hist if start_str <= h.get("date", "") <= end_str]

    # FMP returns descending by date; ensure sorted ascending
    hist_sorted = sorted(hist, key=lambda x: x.get("date", ""))
    return hist_sorted


def compute_post_return(symbol: str, call_date: str, days: int = 3) -> Dict[str, Optional[float]]:
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

    # if call date is in the future, skip price calc
    if call_dt > datetime.utcnow():
        return {"return": None}

    start = call_dt + timedelta(days=0)
    # broaden buffer to catch weekends/holidays; allow up to ~10 extra calendar days
    end = call_dt + timedelta(days=days + 10)
    prices = _historical_prices(symbol, start, end)
    if not prices:
        return {"return": None}

    # find first trading day after call_date
    start_row = next((p for p in prices if p.get("date") > call_dt.strftime("%Y-%m-%d")), None)
    if not start_row:
        return {"return": None}
    start_idx = prices.index(start_row)
    end_idx = min(start_idx + days, len(prices) - 1)
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
    client = _get_client()
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
    def _calendar_from_date(date_str: str) -> Dict[str, Optional[int]]:
        try:
            dt = datetime.fromisoformat(date_str[:10])
            quarter = (dt.month - 1) // 3 + 1
            return {"calendar_year": dt.year, "calendar_quarter": quarter}
        except Exception:
            return {"calendar_year": None, "calendar_quarter": None}

    if not symbol:
        return []

    _require_api_key()
    client = _get_client()
    resp = _get(client, "earning-call-transcript-dates", params={"symbol": symbol, "apikey": FMP_API_KEY})
    resp.raise_for_status()
    data = resp.json() or []

    normalized: List[Dict] = []
    for item in data:
        year = item.get("fiscalYear") or item.get("year")
        quarter = item.get("fiscalQuarter") or item.get("quarter")
        date = item.get("date") or item.get("reportDate")
        cal = _calendar_from_date(date) if date else {"calendar_year": None, "calendar_quarter": None}
        normalized.append(
            {
                "year": year,
                "quarter": quarter,
                "date": date,
                "calendar_year": cal["calendar_year"],
                "calendar_quarter": cal["calendar_quarter"],
            }
        )
    return normalized


def get_transcript(symbol: str, year: int, quarter: int) -> Dict:
    """
    Call FMP Earnings Transcript API.
    """
    _require_api_key()
    client = _get_client()
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
    client = _get_client()
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

    calendar_year = None
    calendar_quarter = None
    if transcript.get("date"):
        try:
            dt = datetime.fromisoformat(transcript["date"][:10])
            calendar_year = dt.year
            calendar_quarter = (dt.month - 1) // 3 + 1
        except Exception:
            pass

    return {
        "symbol": symbol,
        "year": year,
        "quarter": quarter,
        "company": profile.get("company"),
        "sector": profile.get("sector"),
        "exchange": profile.get("exchange"),
        "transcript_text": transcript.get("content", ""),
        "transcript_date": transcript.get("date"),
        "calendar_year": calendar_year,
        "calendar_quarter": calendar_quarter,
        "financials": financials,
        "price_window": price_window,
        "post_earnings_return": post_earnings_return,
        "post_return_meta": post_earnings,
    }
