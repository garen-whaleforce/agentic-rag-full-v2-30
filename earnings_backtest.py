"""
Earnings Backtest Service

計算 earnings call 前後的股價變化：
- BMO (Before Market Open): 前一交易日收盤 → 當天收盤
- AMC (After Market Close): 當天收盤 → 下一交易日收盤
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List, Literal, Optional, TypedDict

logger = logging.getLogger(__name__)

# Type definitions
EarningsSession = Literal["BMO", "AMC", "UNKNOWN"]


class EarningsBacktest(TypedDict, total=False):
    """Earnings backtest result structure."""
    earnings_date: str          # 'YYYY-MM-DD'
    session: EarningsSession    # BMO, AMC, or UNKNOWN
    from_date: str              # 比較起始日
    to_date: str                # 比較結束日
    from_close: Optional[float] # 起始收盤價
    to_close: Optional[float]   # 結束收盤價
    change_pct: Optional[float] # 漲跌百分比 (e.g., -0.67 means -0.67%)


def get_session_from_aws_db(symbol: str, year: int, quarter: int) -> EarningsSession:
    """
    從 AWS DB 的 market_timing 欄位取得 BMO/AMC。

    AWS DB values: 'before_market', 'after_market', 'undetermine'

    Returns: 'BMO', 'AMC', or 'UNKNOWN'
    """
    try:
        from aws_fmp_db import get_market_timing
        timing = get_market_timing(symbol, year, quarter)
        if timing:
            timing_lower = timing.lower()
            if "before" in timing_lower:
                return "BMO"
            elif "after" in timing_lower:
                return "AMC"
    except Exception:
        pass
    return "UNKNOWN"


def infer_earnings_session_from_transcript(transcript: str) -> EarningsSession:
    """
    從 transcript 內容推斷 earnings call 是盤前 (BMO) 還是盤後 (AMC)。

    注意：此函數已被 get_session_from_aws_db() 取代，保留作為 fallback。

    規則：
    1. 把 transcript 轉成小寫，掃描全文
    2. 找 "good morning", "good afternoon", "good evening" 第一次出現的位置
    3. 取三者中 index 最小且非 -1 的作為判定依據：
       - good morning → BMO (盤前)
       - good afternoon / good evening → AMC (盤後)
    4. 如果都沒出現 → UNKNOWN
    """
    if not transcript:
        return "UNKNOWN"

    text_lower = transcript.lower()

    # Find first occurrence of each greeting
    idx_morning = text_lower.find("good morning")
    idx_afternoon = text_lower.find("good afternoon")
    idx_evening = text_lower.find("good evening")

    # Collect valid (non -1) indices with their session type
    candidates: List[tuple[int, EarningsSession]] = []
    if idx_morning != -1:
        candidates.append((idx_morning, "BMO"))
    if idx_afternoon != -1:
        candidates.append((idx_afternoon, "AMC"))
    if idx_evening != -1:
        candidates.append((idx_evening, "AMC"))

    if not candidates:
        return "UNKNOWN"

    # Return session type of the earliest occurrence
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _get_trading_days_around(
    symbol: str,
    target_date: str,
    days_before: int = 5,
    days_after: int = 5,
) -> List[dict]:
    """
    取得指定日期前後的交易日價格資料。

    複用 fmp_client._historical_prices()。
    """
    # Lazy import to avoid circular dependency
    from fmp_client import _historical_prices

    try:
        dt = datetime.fromisoformat(target_date)
    except ValueError:
        try:
            dt = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            return []

    start = dt - timedelta(days=days_before + 5)  # Extra buffer for weekends/holidays
    end = dt + timedelta(days=days_after + 5)

    return _historical_prices(symbol, start, end)


def get_previous_trading_day(symbol: str, date: str) -> Optional[str]:
    """
    取得指定日期的前一個交易日。

    Returns:
        前一交易日的日期字串 (YYYY-MM-DD)，若找不到回傳 None
    """
    prices = _get_trading_days_around(symbol, date, days_before=10, days_after=0)
    if not prices:
        return None

    # Find the last trading day before the target date
    for p in reversed(prices):
        if p.get("date", "") < date:
            return p.get("date")

    return None


def get_next_trading_day(symbol: str, date: str) -> Optional[str]:
    """
    取得指定日期的下一個交易日。

    Returns:
        下一交易日的日期字串 (YYYY-MM-DD)，若找不到回傳 None
    """
    prices = _get_trading_days_around(symbol, date, days_before=0, days_after=10)
    if not prices:
        return None

    # Find the first trading day after the target date
    for p in prices:
        if p.get("date", "") > date:
            return p.get("date")

    return None


def get_close_price(symbol: str, date: str) -> Optional[float]:
    """
    取得指定 symbol 在指定日期的收盤價。

    Returns:
        收盤價，若該日無資料回傳 None
    """
    prices = _get_trading_days_around(symbol, date, days_before=3, days_after=3)
    if not prices:
        return None

    # Find exact date match
    for p in prices:
        if p.get("date") == date:
            try:
                return float(p.get("close"))
            except (TypeError, ValueError):
                return None

    return None


def compute_earnings_backtest(
    symbol: str,
    earnings_date: str,
    transcript: str,
    year: Optional[int] = None,
    quarter: Optional[int] = None,
) -> Optional[EarningsBacktest]:
    """
    計算 earnings call 的回測資訊。

    Args:
        symbol: 股票代號
        earnings_date: Earnings call 日期 (YYYY-MM-DD)
        transcript: Earnings call 全文
        year: 財報年度 (用於查詢 AWS DB market_timing)
        quarter: 財報季度 (用於查詢 AWS DB market_timing)

    Returns:
        EarningsBacktest 結構，包含：
        - earnings_date: 原始日期
        - session: BMO/AMC/UNKNOWN
        - from_date/to_date: 比較的兩個日期
        - from_close/to_close: 兩個日期的收盤價
        - change_pct: 漲跌百分比
    """
    if not symbol or not earnings_date:
        return None

    try:
        # 檢查日期格式
        dt = datetime.fromisoformat(earnings_date[:10])
        earnings_date = dt.strftime("%Y-%m-%d")
    except ValueError:
        logger.warning(f"Invalid earnings_date format: {earnings_date}")
        return None

    # 1. 優先從 AWS DB 取得 session，若失敗則 fallback 到 transcript 解析
    session: EarningsSession = "UNKNOWN"
    if year and quarter:
        session = get_session_from_aws_db(symbol, year, quarter)

    # Fallback to transcript parsing if AWS DB returns UNKNOWN
    if session == "UNKNOWN" and transcript:
        session = infer_earnings_session_from_transcript(transcript)

    # 2. 根據 session 決定比較日期
    if session == "BMO":
        # 盤前：前一交易日收盤 → 當天收盤
        from_date = get_previous_trading_day(symbol, earnings_date)
        to_date = earnings_date

        # 如果 earnings_date 不是交易日，找最近的交易日
        if from_date and not get_close_price(symbol, to_date):
            # 嘗試找 earnings_date 當天或之後最近的交易日
            next_day = get_next_trading_day(symbol, from_date)
            if next_day and next_day >= earnings_date:
                to_date = next_day
    else:
        # 盤後或 UNKNOWN：當天收盤 → 下一交易日收盤
        from_date = earnings_date
        to_date = get_next_trading_day(symbol, earnings_date)

        # 如果 earnings_date 不是交易日，調整
        if not get_close_price(symbol, from_date):
            prev_day = get_previous_trading_day(symbol, earnings_date)
            if prev_day:
                from_date = prev_day
                to_date = get_next_trading_day(symbol, from_date)

    if not from_date or not to_date:
        logger.warning(f"Could not determine trading days for {symbol} around {earnings_date}")
        return EarningsBacktest(
            earnings_date=earnings_date,
            session=session,
            from_date=from_date or "",
            to_date=to_date or "",
            from_close=None,
            to_close=None,
            change_pct=None,
        )

    # 3. 取得收盤價
    from_close = get_close_price(symbol, from_date)
    to_close = get_close_price(symbol, to_date)

    # 4. 計算漲跌百分比
    change_pct: Optional[float] = None
    if from_close is not None and to_close is not None and from_close != 0:
        change_pct = round((to_close - from_close) / from_close * 100, 2)

    return EarningsBacktest(
        earnings_date=earnings_date,
        session=session,
        from_date=from_date,
        to_date=to_date,
        from_close=round(from_close, 2) if from_close is not None else None,
        to_close=round(to_close, 2) if to_close is not None else None,
        change_pct=change_pct,
    )
