"""
AWS FMP Database Client

Provides access to FMP data cached in AWS PostgreSQL (pead_reversal database).
Used as primary data source before falling back to FMP API.

Tables:
- companies: symbol, name, sector, sub_sector
- earnings_transcripts: metadata (company_id, symbol, year, quarter, transcript_date)
- transcript_content: full transcript text
- income_statements, balance_sheets, cash_flow_statements: financial data
- historical_prices: daily OHLCV data
- price_analysis: pre-computed price analysis around earnings
"""

import os
import logging
from typing import Dict, List, Optional
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

AWS_FMP_DSN = os.getenv("AWS_FMP_DSN", "postgresql://postgres:password@127.0.0.1:15432/pead_reversal")

_conn = None


def _get_connection():
    """Get or create a database connection."""
    global _conn
    if _conn is None or _conn.closed:
        try:
            _conn = psycopg2.connect(AWS_FMP_DSN)
            _conn.autocommit = True
        except Exception as e:
            logger.warning("Failed to connect to AWS FMP DB: %s", e)
            return None
    return _conn


def close_connection():
    """Close the database connection."""
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:
            pass
        _conn = None


@contextmanager
def get_cursor():
    """Context manager for database cursor."""
    conn = _get_connection()
    if conn is None:
        yield None
        return
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        yield cur
        cur.close()
    except Exception as e:
        logger.warning("Database cursor error: %s", e)
        yield None


def get_company_profile(symbol: str) -> Optional[Dict]:
    """
    Fetch company profile from AWS database.

    Returns: Dict with company, sector, sub_sector or None if not found.
    """
    if not symbol:
        return None

    with get_cursor() as cur:
        if cur is None:
            return None
        try:
            cur.execute("""
                SELECT symbol, name, sector, sub_sector
                FROM companies
                WHERE UPPER(symbol) = %s
            """, (symbol.upper(),))
            row = cur.fetchone()
            if row:
                return {
                    "symbol": row["symbol"],
                    "company": row["name"],
                    "sector": row["sector"],
                    "sub_sector": row.get("sub_sector"),
                }
        except Exception as e:
            logger.debug("get_company_profile error: %s", e)
    return None


def get_peers_by_sector(sector: str, exclude_symbol: str = None, limit: int = 10) -> List[str]:
    """
    Get peer company symbols in the same sector.

    Returns: List of ticker symbols.
    """
    if not sector:
        return []

    with get_cursor() as cur:
        if cur is None:
            return []
        try:
            if exclude_symbol:
                cur.execute("""
                    SELECT symbol FROM companies
                    WHERE sector = %s AND UPPER(symbol) != %s
                    LIMIT %s
                """, (sector, exclude_symbol.upper(), limit))
            else:
                cur.execute("""
                    SELECT symbol FROM companies
                    WHERE sector = %s
                    LIMIT %s
                """, (sector, limit))
            return [row["symbol"] for row in cur.fetchall()]
        except Exception as e:
            logger.debug("get_peers_by_sector error: %s", e)
    return []


def get_transcript(symbol: str, year: int, quarter: int) -> Optional[Dict]:
    """
    Fetch earnings transcript from AWS database.

    Returns: Dict with symbol, year, quarter, date, content or None if not found.
    """
    if not symbol or not year or not quarter:
        return None

    with get_cursor() as cur:
        if cur is None:
            return None
        try:
            cur.execute("""
                SELECT
                    tc.symbol, tc.year, tc.quarter, tc.content,
                    et.transcript_date
                FROM transcript_content tc
                JOIN earnings_transcripts et ON tc.transcript_id = et.id
                WHERE UPPER(tc.symbol) = %s AND tc.year = %s AND tc.quarter = %s
            """, (symbol.upper(), year, quarter))
            row = cur.fetchone()
            if row and row.get("content"):
                return {
                    "symbol": row["symbol"],
                    "year": row["year"],
                    "quarter": row["quarter"],
                    "date": str(row["transcript_date"]) if row.get("transcript_date") else None,
                    "content": row["content"],
                }
        except Exception as e:
            logger.debug("get_transcript error: %s", e)
    return None


def get_transcript_dates(symbol: str) -> List[Dict]:
    """
    Get all available transcript dates for a symbol.

    Returns: List of dicts with year, quarter, date.
    """
    if not symbol:
        return []

    with get_cursor() as cur:
        if cur is None:
            return []
        try:
            cur.execute("""
                SELECT year, quarter, transcript_date
                FROM earnings_transcripts
                WHERE UPPER(symbol) = %s
                ORDER BY year DESC, quarter DESC
            """, (symbol.upper(),))
            results = []
            for row in cur.fetchall():
                results.append({
                    "year": row["year"],
                    "quarter": row["quarter"],
                    "date": str(row["transcript_date"]) if row.get("transcript_date") else None,
                })
            return results
        except Exception as e:
            logger.debug("get_transcript_dates error: %s", e)
    return []


def get_quarterly_financials(symbol: str, limit: int = 4) -> Optional[Dict]:
    """
    Fetch quarterly financial statements from AWS database.

    Returns: Dict with income, balance, cashFlow arrays or None if not found.
    """
    if not symbol:
        return None

    with get_cursor() as cur:
        if cur is None:
            return None
        try:
            # Income statements
            cur.execute("""
                SELECT * FROM income_statements
                WHERE UPPER(symbol) = %s
                ORDER BY date DESC
                LIMIT %s
            """, (symbol.upper(), limit))
            income = [dict(row) for row in cur.fetchall()]

            # Balance sheets
            cur.execute("""
                SELECT * FROM balance_sheets
                WHERE UPPER(symbol) = %s
                ORDER BY date DESC
                LIMIT %s
            """, (symbol.upper(), limit))
            balance = [dict(row) for row in cur.fetchall()]

            # Cash flow statements
            cur.execute("""
                SELECT * FROM cash_flow_statements
                WHERE UPPER(symbol) = %s
                ORDER BY date DESC
                LIMIT %s
            """, (symbol.upper(), limit))
            cash_flow = [dict(row) for row in cur.fetchall()]

            if income or balance or cash_flow:
                return {
                    "income": income,
                    "balance": balance,
                    "cashFlow": cash_flow,
                }
        except Exception as e:
            logger.debug("get_quarterly_financials error: %s", e)
    return None


def get_historical_prices(symbol: str, start_date: str, end_date: str) -> List[Dict]:
    """
    Fetch historical prices from AWS database.

    Args:
        symbol: Ticker symbol
        start_date: YYYY-MM-DD format
        end_date: YYYY-MM-DD format

    Returns: List of price dicts sorted by date ascending.
    """
    if not symbol:
        return []

    with get_cursor() as cur:
        if cur is None:
            return []
        try:
            cur.execute("""
                SELECT date, open, high, low, close, volume
                FROM historical_prices
                WHERE UPPER(symbol) = %s AND date >= %s AND date <= %s
                ORDER BY date ASC
            """, (symbol.upper(), start_date, end_date))
            results = []
            for row in cur.fetchall():
                results.append({
                    "date": str(row["date"]),
                    "open": float(row["open"]) if row.get("open") else None,
                    "high": float(row["high"]) if row.get("high") else None,
                    "low": float(row["low"]) if row.get("low") else None,
                    "close": float(row["close"]) if row.get("close") else None,
                    "volume": int(row["volume"]) if row.get("volume") else None,
                })
            return results
        except Exception as e:
            logger.debug("get_historical_prices error: %s", e)
    return []


def get_price_analysis(symbol: str, year: int, quarter: int) -> Optional[Dict]:
    """
    Get pre-computed price analysis for an earnings transcript.

    Returns: Dict with t_day, price_t_minus_1, price_t, etc. or None.
    """
    if not symbol:
        return None

    with get_cursor() as cur:
        if cur is None:
            return None
        try:
            cur.execute("""
                SELECT pa.*
                FROM price_analysis pa
                JOIN earnings_transcripts et ON pa.transcript_id = et.id
                WHERE UPPER(et.symbol) = %s AND et.year = %s AND et.quarter = %s
            """, (symbol.upper(), year, quarter))
            row = cur.fetchone()
            if row:
                return dict(row)
        except Exception as e:
            logger.debug("get_price_analysis error: %s", e)
    return None


def check_connection() -> bool:
    """Test if database connection is working."""
    with get_cursor() as cur:
        if cur is None:
            return False
        try:
            cur.execute("SELECT 1")
            return True
        except Exception:
            return False


def get_all_sectors() -> List[str]:
    """Get list of all unique sectors in the database."""
    with get_cursor() as cur:
        if cur is None:
            return []
        try:
            cur.execute("""
                SELECT DISTINCT sector FROM companies
                WHERE sector IS NOT NULL
                ORDER BY sector
            """)
            return [row["sector"] for row in cur.fetchall()]
        except Exception as e:
            logger.debug("get_all_sectors error: %s", e)
    return []


def get_companies_count() -> int:
    """Get total number of companies in database."""
    with get_cursor() as cur:
        if cur is None:
            return 0
        try:
            cur.execute("SELECT COUNT(*) as cnt FROM companies")
            row = cur.fetchone()
            return row["cnt"] if row else 0
        except Exception:
            return 0
