"""aws_db_client.py - AWS PostgreSQL (pead_reversal) 歷史數據客戶端

此模組連接 AWS PostgreSQL 數據庫，提供 S&P 500 2021-2025 的歷史數據：
- Earnings Call Transcripts
- Financial Statements (Income, Balance Sheet, Cash Flow)
- Historical Prices
- GICS Classification
- Post-Earnings Price Analysis
"""

from __future__ import annotations

import os
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

AWS_PG_DSN = os.getenv("AWS_POSTGRES_DSN")

# Connection pool (simple implementation)
_conn: Optional[psycopg2.extensions.connection] = None


def _get_conn() -> psycopg2.extensions.connection:
    """Get or create a connection to AWS PostgreSQL."""
    global _conn
    if _conn is None or _conn.closed:
        if not AWS_PG_DSN:
            raise RuntimeError("AWS_POSTGRES_DSN not set in environment")
        _conn = psycopg2.connect(AWS_PG_DSN, cursor_factory=RealDictCursor)
    return _conn


def _ensure_conn() -> bool:
    """Check if AWS DB is available."""
    if not AWS_PG_DSN:
        return False
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:
        return False


def is_available() -> bool:
    """Check if AWS DB connection is available."""
    return _ensure_conn()


# =============================================================================
# Transcript Functions
# =============================================================================

def get_transcript(symbol: str, year: int, quarter: int) -> Optional[str]:
    """Get earnings call transcript content from AWS DB.

    Args:
        symbol: Stock ticker (e.g., 'AAPL')
        year: Fiscal year (e.g., 2024)
        quarter: Fiscal quarter (1-4)

    Returns:
        Transcript content as string, or None if not found
    """
    if not AWS_PG_DSN:
        return None
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT content
                FROM transcript_content
                WHERE symbol = %s AND year = %s AND quarter = %s
                LIMIT 1
            """, (symbol.upper(), year, quarter))
            row = cur.fetchone()
            return row["content"] if row else None
    except Exception as e:
        print(f"[AWS DB] get_transcript error: {e}")
        return None


def get_transcript_metadata(symbol: str, year: int, quarter: int) -> Optional[Dict[str, Any]]:
    """Get earnings call metadata from AWS DB.

    Returns dict with: transcript_date, market_timing, t_day, etc.
    """
    if not AWS_PG_DSN:
        return None
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    symbol, year, quarter,
                    transcript_date, transcript_date_str,
                    t_day, market_timing, detection_method
                FROM earnings_transcripts
                WHERE symbol = %s AND year = %s AND quarter = %s
                LIMIT 1
            """, (symbol.upper(), year, quarter))
            row = cur.fetchone()
            return _convert_decimals(dict(row)) if row else None
    except Exception as e:
        print(f"[AWS DB] get_transcript_metadata error: {e}")
        return None


def get_all_transcript_dates(symbol: str) -> List[Dict[str, Any]]:
    """Get all earnings call dates for a symbol."""
    if not AWS_PG_DSN:
        return []
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT year, quarter, transcript_date_str, t_day, market_timing
                FROM earnings_transcripts
                WHERE symbol = %s
                ORDER BY year DESC, quarter DESC
            """, (symbol.upper(),))
            return [_convert_decimals(dict(row)) for row in cur.fetchall()]
    except Exception as e:
        print(f"[AWS DB] get_all_transcript_dates error: {e}")
        return []


# =============================================================================
# Company / GICS Functions
# =============================================================================

def get_company_info(symbol: str) -> Optional[Dict[str, Any]]:
    """Get company info including GICS classification.

    Returns dict with: name, sector, sub_sector, gics, cik
    """
    if not AWS_PG_DSN:
        return None
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT symbol, name, sector, sub_sector, gics, cik
                FROM companies
                WHERE symbol = %s
                LIMIT 1
            """, (symbol.upper(),))
            row = cur.fetchone()
            return _convert_decimals(dict(row)) if row else None
    except Exception as e:
        print(f"[AWS DB] get_company_info error: {e}")
        return None


def get_companies_by_sector(sector: str) -> List[str]:
    """Get all company symbols in a given sector."""
    if not AWS_PG_DSN:
        return []
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT symbol FROM companies
                WHERE sector = %s OR gics = %s
                ORDER BY symbol
            """, (sector, sector))
            return [row["symbol"] for row in cur.fetchall()]
    except Exception as e:
        print(f"[AWS DB] get_companies_by_sector error: {e}")
        return []


def get_all_companies() -> List[Dict[str, Any]]:
    """Get all companies with their GICS info."""
    if not AWS_PG_DSN:
        return []
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT symbol, name, sector, sub_sector, gics
                FROM companies
                ORDER BY symbol
            """)
            return [_convert_decimals(dict(row)) for row in cur.fetchall()]
    except Exception as e:
        print(f"[AWS DB] get_all_companies error: {e}")
        return []


# =============================================================================
# Financial Statements Functions
# =============================================================================

def _convert_decimals(row: Dict) -> Dict:
    """Convert Decimal and date values for JSON serialization."""
    from datetime import date, datetime
    result = {}
    for k, v in row.items():
        if isinstance(v, Decimal):
            result[k] = float(v)
        elif isinstance(v, (date, datetime)):
            result[k] = v.isoformat() if hasattr(v, 'isoformat') else str(v)
        else:
            result[k] = v
    return result


def get_income_statements(symbol: str, limit: int = 4) -> List[Dict[str, Any]]:
    """Get quarterly income statements."""
    if not AWS_PG_DSN:
        return []
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM income_statements
                WHERE symbol = %s
                ORDER BY date DESC
                LIMIT %s
            """, (symbol.upper(), limit))
            return [_convert_decimals(dict(row)) for row in cur.fetchall()]
    except Exception as e:
        print(f"[AWS DB] get_income_statements error: {e}")
        return []


def get_balance_sheets(symbol: str, limit: int = 4) -> List[Dict[str, Any]]:
    """Get quarterly balance sheets."""
    if not AWS_PG_DSN:
        return []
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM balance_sheets
                WHERE symbol = %s
                ORDER BY date DESC
                LIMIT %s
            """, (symbol.upper(), limit))
            return [_convert_decimals(dict(row)) for row in cur.fetchall()]
    except Exception as e:
        print(f"[AWS DB] get_balance_sheets error: {e}")
        return []


def get_cash_flow_statements(symbol: str, limit: int = 4) -> List[Dict[str, Any]]:
    """Get quarterly cash flow statements."""
    if not AWS_PG_DSN:
        return []
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM cash_flow_statements
                WHERE symbol = %s
                ORDER BY date DESC
                LIMIT %s
            """, (symbol.upper(), limit))
            return [_convert_decimals(dict(row)) for row in cur.fetchall()]
    except Exception as e:
        print(f"[AWS DB] get_cash_flow_statements error: {e}")
        return []


def get_quarterly_financials(symbol: str, limit: int = 4) -> Dict[str, List[Dict]]:
    """Get all three financial statements for a symbol.

    Returns dict with keys: income, balance, cashFlow
    """
    return {
        "income": get_income_statements(symbol, limit),
        "balance": get_balance_sheets(symbol, limit),
        "cashFlow": get_cash_flow_statements(symbol, limit),
    }


# =============================================================================
# Historical Prices Functions
# =============================================================================

def get_historical_prices(
    symbol: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> List[Dict[str, Any]]:
    """Get historical daily prices for a symbol.

    Args:
        symbol: Stock ticker
        start_date: Start date (inclusive)
        end_date: End date (inclusive)

    Returns:
        List of price records with: date, open, high, low, close, adj_close, volume
    """
    if not AWS_PG_DSN:
        return []
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            query = "SELECT date, open, high, low, close, adj_close, volume FROM historical_prices WHERE symbol = %s"
            params: List[Any] = [symbol.upper()]

            if start_date:
                query += " AND date >= %s"
                params.append(start_date)
            if end_date:
                query += " AND date <= %s"
                params.append(end_date)

            query += " ORDER BY date DESC"
            cur.execute(query, params)
            return [_convert_decimals(dict(row)) for row in cur.fetchall()]
    except Exception as e:
        print(f"[AWS DB] get_historical_prices error: {e}")
        return []


def get_price_on_date(symbol: str, target_date: date) -> Optional[Dict[str, Any]]:
    """Get price data for a specific date."""
    if not AWS_PG_DSN:
        return None
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT date, open, high, low, close, adj_close, volume
                FROM historical_prices
                WHERE symbol = %s AND date = %s
                LIMIT 1
            """, (symbol.upper(), target_date))
            row = cur.fetchone()
            return _convert_decimals(dict(row)) if row else None
    except Exception as e:
        print(f"[AWS DB] get_price_on_date error: {e}")
        return None


# =============================================================================
# Price Analysis Functions (Post-Earnings Returns)
# =============================================================================

def get_price_analysis(symbol: str, year: int, quarter: int) -> Optional[Dict[str, Any]]:
    """Get post-earnings price analysis for a specific earnings call.

    Returns dict with:
        - t_day: earnings call date
        - price_t: price on earnings day
        - pct_change_t: same-day % change
        - price_t_plus_20/30/40/50: prices N days after
        - pct_change_t_plus_20/30/40/50: % changes
        - trend_category: classification
    """
    if not AWS_PG_DSN:
        return None
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            # Join with earnings_transcripts to match by year/quarter
            cur.execute("""
                SELECT pa.*
                FROM price_analysis pa
                JOIN earnings_transcripts et ON pa.transcript_id = et.id
                WHERE et.symbol = %s AND et.year = %s AND et.quarter = %s
                LIMIT 1
            """, (symbol.upper(), year, quarter))
            row = cur.fetchone()
            return _convert_decimals(dict(row)) if row else None
    except Exception as e:
        print(f"[AWS DB] get_price_analysis error: {e}")
        return None


def get_post_earnings_return(symbol: str, year: int, quarter: int, days: int = 1) -> Optional[float]:
    """Get post-earnings return for a specific period.

    Args:
        symbol: Stock ticker
        year: Fiscal year
        quarter: Fiscal quarter
        days: Number of days after earnings (1, 20, 30, 40, or 50)

    Returns:
        Percentage return as float (e.g., 5.5 for +5.5%), or None
    """
    analysis = get_price_analysis(symbol, year, quarter)
    if not analysis:
        return None

    if days == 1:
        return analysis.get("pct_change_t")

    field = f"pct_change_t_plus_{days}"
    return analysis.get(field)


# =============================================================================
# Utility Functions
# =============================================================================

def close_connection():
    """Close the AWS DB connection."""
    global _conn
    if _conn and not _conn.closed:
        _conn.close()
        _conn = None


# =============================================================================
# Peer Comparison Functions
# =============================================================================

def get_peer_transcripts(
    symbol: str,
    quarter: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """Get peer company transcripts from same sector for a given quarter.

    Args:
        symbol: Stock ticker (e.g., 'AAPL')
        quarter: Quarter string (e.g., '2024-Q3')
        limit: Max number of peer transcripts to return

    Returns:
        List of dicts with: symbol, name, sector, transcript content
    """
    if not AWS_PG_DSN:
        return []
    try:
        # Parse quarter string (e.g., '2024-Q3' -> year=2024, quarter=3)
        parts = quarter.split('-Q')
        if len(parts) != 2:
            return []
        year = int(parts[0])
        q = int(parts[1])

        conn = _get_conn()
        with conn.cursor() as cur:
            # Get company sector first
            cur.execute("""
                SELECT sector FROM companies WHERE symbol = %s LIMIT 1
            """, (symbol.upper(),))
            row = cur.fetchone()
            if not row or not row.get("sector"):
                return []
            sector = row["sector"]

            # Get peer transcripts from same sector
            cur.execute("""
                SELECT c.symbol, c.name, c.sector, tc.content
                FROM companies c
                JOIN earnings_transcripts et ON c.symbol = et.symbol
                JOIN transcript_content tc ON et.symbol = tc.symbol
                    AND et.year = tc.year AND et.quarter = tc.quarter
                WHERE c.sector = %s
                    AND c.symbol != %s
                    AND et.year = %s AND et.quarter = %s
                ORDER BY c.symbol
                LIMIT %s
            """, (sector, symbol.upper(), year, q, limit))
            return [_convert_decimals(dict(row)) for row in cur.fetchall()]
    except Exception as e:
        print(f"[AWS DB] get_peer_transcripts error: {e}")
        return []


def get_peer_financials(
    symbol: str,
    quarter: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """Get peer company financials from same sector.

    Args:
        symbol: Stock ticker
        quarter: Quarter string (e.g., '2024-Q3')
        limit: Max number of peers

    Returns:
        List of dicts with: symbol, name, sector, income, balance, cashFlow
    """
    if not AWS_PG_DSN:
        return []
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            # Get company sector
            cur.execute("""
                SELECT sector FROM companies WHERE symbol = %s LIMIT 1
            """, (symbol.upper(),))
            row = cur.fetchone()
            if not row or not row.get("sector"):
                return []
            sector = row["sector"]

            # Get peer symbols
            cur.execute("""
                SELECT symbol, name, sector FROM companies
                WHERE sector = %s AND symbol != %s
                ORDER BY symbol
                LIMIT %s
            """, (sector, symbol.upper(), limit))
            peers = [_convert_decimals(dict(row)) for row in cur.fetchall()]

            # Get financials for each peer
            results = []
            for peer in peers:
                peer_symbol = peer["symbol"]
                peer["income"] = get_income_statements(peer_symbol, 2)
                peer["balance"] = get_balance_sheets(peer_symbol, 2)
                peer["cashFlow"] = get_cash_flow_statements(peer_symbol, 2)
                results.append(peer)

            return results
    except Exception as e:
        print(f"[AWS DB] get_peer_financials error: {e}")
        return []


def get_peer_facts_summary(
    symbol: str,
    quarter: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """Get summarized peer data combining transcripts and key financials.

    This provides a quick peer comparison without full transcript content.

    Args:
        symbol: Stock ticker
        quarter: Quarter string (e.g., '2024-Q3')
        limit: Max number of peers

    Returns:
        List of dicts with peer info and key metrics
    """
    if not AWS_PG_DSN:
        return []
    try:
        parts = quarter.split('-Q')
        if len(parts) != 2:
            return []
        year = int(parts[0])
        q = int(parts[1])

        conn = _get_conn()
        with conn.cursor() as cur:
            # Get company sector
            cur.execute("""
                SELECT sector FROM companies WHERE symbol = %s LIMIT 1
            """, (symbol.upper(),))
            row = cur.fetchone()
            if not row or not row.get("sector"):
                return []
            sector = row["sector"]

            # Get peer data with key financial metrics
            cur.execute("""
                SELECT
                    c.symbol, c.name, c.sector,
                    inc.revenue, inc.net_income, inc.eps,
                    inc.revenue_growth, inc.operating_income,
                    pa.pct_change_t as earnings_day_return,
                    pa.pct_change_t_plus_20 as return_20d
                FROM companies c
                LEFT JOIN income_statements inc ON c.symbol = inc.symbol
                LEFT JOIN earnings_transcripts et ON c.symbol = et.symbol
                    AND et.year = %s AND et.quarter = %s
                LEFT JOIN price_analysis pa ON et.id = pa.transcript_id
                WHERE c.sector = %s AND c.symbol != %s
                ORDER BY inc.date DESC, c.symbol
                LIMIT %s
            """, (year, q, sector, symbol.upper(), limit))

            # Dedupe by symbol (keep most recent)
            seen = set()
            results = []
            for row in cur.fetchall():
                if row["symbol"] not in seen:
                    seen.add(row["symbol"])
                    results.append(_convert_decimals(dict(row)))

            return results
    except Exception as e:
        print(f"[AWS DB] get_peer_facts_summary error: {e}")
        return []


# =============================================================================
# Historical Performance Functions (for HistoricalPerformanceAgent)
# =============================================================================

def get_historical_financials_summary(
    symbol: str,
    current_quarter: str,
    num_quarters: int = 8
) -> List[Dict[str, Any]]:
    """Get historical financial metrics for a company (past N quarters).

    Used by HistoricalPerformanceAgent to compare current facts with past performance.

    Args:
        symbol: Stock ticker
        current_quarter: Current quarter string (e.g., '2024-Q3') to exclude
        num_quarters: Number of past quarters to fetch

    Returns:
        List of dicts with: quarter, revenue, net_income, eps, revenue_growth, etc.
    """
    if not AWS_PG_DSN:
        return []
    try:
        # Parse current quarter to filter it out
        parts = current_quarter.split('-Q')
        if len(parts) != 2:
            return []
        current_year = int(parts[0])
        current_q = int(parts[1])

        conn = _get_conn()
        with conn.cursor() as cur:
            # Get historical income statements (past quarters only)
            cur.execute("""
                SELECT
                    inc.date, inc.period,
                    inc.revenue, inc.net_income, inc.eps, inc.ebitda,
                    inc.revenue_growth, inc.gross_profit, inc.operating_income,
                    inc.cost_of_revenue, inc.operating_expenses
                FROM income_statements inc
                WHERE inc.symbol = %s
                ORDER BY inc.date DESC
                LIMIT %s
            """, (symbol.upper(), num_quarters + 2))  # +2 to ensure we have enough after filtering

            results = []
            for row in cur.fetchall():
                row_dict = _convert_decimals(dict(row))
                # Infer quarter from date if period not available
                if row_dict.get("date"):
                    d = row_dict["date"]
                    year = d.year if hasattr(d, 'year') else int(str(d)[:4])
                    month = d.month if hasattr(d, 'month') else int(str(d)[5:7])
                    q = (month - 1) // 3 + 1
                    row_dict["quarter"] = f"{year}-Q{q}"

                    # Skip current quarter
                    if year == current_year and q == current_q:
                        continue

                results.append(row_dict)

            return results[:num_quarters]
    except Exception as e:
        print(f"[AWS DB] get_historical_financials_summary error: {e}")
        return []


def get_historical_financials_facts(
    symbol: str,
    current_quarter: str,
    num_quarters: int = 8
) -> List[Dict[str, Any]]:
    """Get historical financial data formatted as facts for LLM comparison.

    Returns facts in format compatible with HistoricalPerformanceAgent.

    Args:
        symbol: Stock ticker
        current_quarter: Current quarter to exclude
        num_quarters: Number of past quarters

    Returns:
        List of fact dicts with: metric, value, reason, quarter, ticker
    """
    financials = get_historical_financials_summary(symbol, current_quarter, num_quarters)
    if not financials:
        return []

    facts = []
    for fin in financials:
        quarter = fin.get("quarter", "Unknown")

        if fin.get("revenue"):
            facts.append({
                "metric": "Revenue",
                "value": f"${fin['revenue']:,.0f}",
                "reason": f"Historical quarterly revenue",
                "quarter": quarter,
                "ticker": symbol.upper(),
                "type": "Result",
            })
        if fin.get("net_income"):
            facts.append({
                "metric": "Net Income",
                "value": f"${fin['net_income']:,.0f}",
                "reason": f"Historical quarterly net income",
                "quarter": quarter,
                "ticker": symbol.upper(),
                "type": "Result",
            })
        if fin.get("eps"):
            facts.append({
                "metric": "EPS",
                "value": f"${fin['eps']:.2f}",
                "reason": f"Historical quarterly EPS",
                "quarter": quarter,
                "ticker": symbol.upper(),
                "type": "Result",
            })
        if fin.get("revenue_growth"):
            facts.append({
                "metric": "Revenue Growth",
                "value": f"{fin['revenue_growth']:.1%}" if isinstance(fin['revenue_growth'], float) else str(fin['revenue_growth']),
                "reason": f"Historical YoY revenue growth",
                "quarter": quarter,
                "ticker": symbol.upper(),
                "type": "Result",
            })
        if fin.get("gross_profit"):
            facts.append({
                "metric": "Gross Profit",
                "value": f"${fin['gross_profit']:,.0f}",
                "reason": f"Historical gross profit",
                "quarter": quarter,
                "ticker": symbol.upper(),
                "type": "Result",
            })
        if fin.get("operating_income"):
            facts.append({
                "metric": "Operating Income",
                "value": f"${fin['operating_income']:,.0f}",
                "reason": f"Historical operating income",
                "quarter": quarter,
                "ticker": symbol.upper(),
                "type": "Result",
            })

    return facts


# =============================================================================
# Historical Earnings Functions (for HistoricalEarningsAgent)
# =============================================================================

def get_historical_earnings_summary(
    symbol: str,
    current_quarter: str,
    num_quarters: int = 8
) -> List[Dict[str, Any]]:
    """Get historical earnings call data for a company.

    Used by HistoricalEarningsAgent to compare with past earnings calls.

    Args:
        symbol: Stock ticker
        current_quarter: Current quarter string to exclude
        num_quarters: Number of past quarters

    Returns:
        List of dicts with: year, quarter, transcript_date, market_timing, price analysis
    """
    if not AWS_PG_DSN:
        return []
    try:
        parts = current_quarter.split('-Q')
        if len(parts) != 2:
            return []
        current_year = int(parts[0])
        current_q = int(parts[1])

        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    et.year, et.quarter, et.transcript_date_str,
                    et.market_timing,
                    pa.pct_change_t as earnings_day_return,
                    pa.pct_change_t_plus_20 as return_20d,
                    pa.pct_change_t_plus_30 as return_30d,
                    pa.trend_category
                FROM earnings_transcripts et
                LEFT JOIN price_analysis pa ON et.id = pa.transcript_id
                WHERE et.symbol = %s
                    AND NOT (et.year = %s AND et.quarter = %s)
                ORDER BY et.year DESC, et.quarter DESC
                LIMIT %s
            """, (symbol.upper(), current_year, current_q, num_quarters))

            results = []
            for row in cur.fetchall():
                row_dict = _convert_decimals(dict(row))
                row_dict["quarter_str"] = f"{row_dict['year']}-Q{row_dict['quarter']}"
                results.append(row_dict)

            return results
    except Exception as e:
        print(f"[AWS DB] get_historical_earnings_summary error: {e}")
        return []


def get_historical_earnings_facts(
    symbol: str,
    current_quarter: str,
    num_quarters: int = 8
) -> List[Dict[str, Any]]:
    """Get historical earnings data formatted as facts for LLM comparison.

    Returns facts in format compatible with HistoricalEarningsAgent.

    Args:
        symbol: Stock ticker
        current_quarter: Current quarter to exclude
        num_quarters: Number of past quarters

    Returns:
        List of fact dicts with: metric, value, reason, quarter, ticker
    """
    earnings = get_historical_earnings_summary(symbol, current_quarter, num_quarters)
    if not earnings:
        return []

    facts = []
    for earn in earnings:
        quarter = earn.get("quarter_str", "Unknown")

        if earn.get("earnings_day_return") is not None:
            facts.append({
                "metric": "Earnings Day Return",
                "value": f"{earn['earnings_day_return']:.2f}%",
                "reason": f"Post-earnings price movement on announcement day",
                "quarter": quarter,
                "ticker": symbol.upper(),
                "type": "Result",
            })
        if earn.get("return_20d") is not None:
            facts.append({
                "metric": "20-Day Post-Earnings Return",
                "value": f"{earn['return_20d']:.2f}%",
                "reason": f"Price movement 20 days after earnings",
                "quarter": quarter,
                "ticker": symbol.upper(),
                "type": "Result",
            })
        if earn.get("trend_category"):
            facts.append({
                "metric": "Post-Earnings Trend",
                "value": earn["trend_category"],
                "reason": f"Classified trend pattern after earnings",
                "quarter": quarter,
                "ticker": symbol.upper(),
                "type": "Result",
            })
        if earn.get("market_timing"):
            facts.append({
                "metric": "Earnings Timing",
                "value": earn["market_timing"],
                "reason": f"When earnings were announced (BMO/AMC)",
                "quarter": quarter,
                "ticker": symbol.upper(),
                "type": "Result",
            })

    return facts


# =============================================================================
# Transcript Dates Functions (for fmp_client optimization)
# =============================================================================

def get_transcript_dates_aws(symbol: str) -> List[Dict[str, Any]]:
    """Get available transcript dates for a symbol from AWS DB.

    Used by fmp_client.get_transcript_dates as primary source.

    Args:
        symbol: Stock ticker

    Returns:
        List of dicts with: year, quarter, transcript_date_str, calendar_year, calendar_quarter
    """
    if not AWS_PG_DSN:
        return []
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    et.year, et.quarter,
                    et.transcript_date_str,
                    et.market_timing
                FROM earnings_transcripts et
                WHERE et.symbol = %s
                ORDER BY et.year DESC, et.quarter DESC
            """, (symbol.upper(),))

            results = []
            for row in cur.fetchall():
                row_dict = _convert_decimals(dict(row))
                # Calculate calendar year/quarter from transcript date
                date_str = row_dict.get("transcript_date_str") or ""
                calendar_year = None
                calendar_quarter = None
                if date_str:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(date_str[:10])
                        calendar_year = dt.year
                        calendar_quarter = (dt.month - 1) // 3 + 1
                    except Exception:
                        pass

                results.append({
                    "year": row_dict.get("year"),
                    "quarter": row_dict.get("quarter"),
                    "date": date_str,
                    "calendar_year": calendar_year,
                    "calendar_quarter": calendar_quarter,
                })

            return results
    except Exception as e:
        print(f"[AWS DB] get_transcript_dates_aws error: {e}")
        return []


def get_fiscal_quarter_by_date_aws(symbol: str, date: str) -> Optional[Dict[str, int]]:
    """Find fiscal year/quarter for a given earnings date from AWS DB.

    Used by fmp_client.get_fiscal_quarter_by_date as primary source.

    Args:
        symbol: Stock ticker
        date: Date string YYYY-MM-DD

    Returns:
        Dict with year, quarter or None if not found
    """
    if not AWS_PG_DSN:
        return None
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            # Try exact match first
            cur.execute("""
                SELECT year, quarter
                FROM earnings_transcripts
                WHERE symbol = %s AND transcript_date_str = %s
                LIMIT 1
            """, (symbol.upper(), date))

            row = cur.fetchone()
            if row:
                return {"year": row["year"], "quarter": row["quarter"]}

            # Try partial match (date starts with)
            cur.execute("""
                SELECT year, quarter
                FROM earnings_transcripts
                WHERE symbol = %s AND transcript_date_str LIKE %s
                LIMIT 1
            """, (symbol.upper(), f"{date}%"))

            row = cur.fetchone()
            if row:
                return {"year": row["year"], "quarter": row["quarter"]}

            return None
    except Exception as e:
        print(f"[AWS DB] get_fiscal_quarter_by_date_aws error: {e}")
        return None


def get_stats() -> Dict[str, int]:
    """Get row counts for all tables."""
    if not AWS_PG_DSN:
        return {}
    try:
        conn = _get_conn()
        stats = {}
        tables = [
            "companies", "earnings_transcripts", "transcript_content",
            "historical_prices", "income_statements", "balance_sheets",
            "cash_flow_statements", "price_analysis"
        ]
        with conn.cursor() as cur:
            for table in tables:
                cur.execute(f"SELECT COUNT(*) as cnt FROM {table}")
                stats[table] = cur.fetchone()["cnt"]
        return stats
    except Exception as e:
        print(f"[AWS DB] get_stats error: {e}")
        return {}
