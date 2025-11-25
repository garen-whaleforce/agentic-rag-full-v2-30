from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = Path(os.getenv("ANALYSIS_DB_PATH", "data/analysis_results.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS call_results (
                job_id TEXT PRIMARY KEY,
                symbol TEXT,
                company TEXT,
                fiscal_year INTEGER,
                fiscal_quarter INTEGER,
                call_date TEXT,
                sector TEXT,
                exchange TEXT,
                post_return REAL,
                prediction TEXT,
                confidence REAL,
                correct INTEGER,
                agent_result_json TEXT,
                token_usage_json TEXT,
                agent_notes TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_call_date ON call_results(call_date)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbol ON call_results(symbol)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_prediction ON call_results(prediction)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS call_cache (
                symbol TEXT NOT NULL,
                fiscal_year INTEGER NOT NULL,
                fiscal_quarter INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (symbol, fiscal_year, fiscal_quarter)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fmp_cache (
                cache_key TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )


def record_analysis(
    job_id: str,
    symbol: str,
    fiscal_year: Optional[int],
    fiscal_quarter: Optional[int],
    call_date: Optional[str],
    sector: Optional[str],
    exchange: Optional[str],
    post_return: Optional[float],
    prediction: Optional[str],
    confidence: Optional[float],
    correct: Optional[bool],
    agent_result: Dict[str, Any],
    token_usage: Optional[Dict[str, Any]],
    agent_notes: Optional[str],
    company: Optional[str] = None,
) -> None:
    init_db()
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO call_results (
                job_id, symbol, company, fiscal_year, fiscal_quarter, call_date, sector, exchange,
                post_return, prediction, confidence, correct, agent_result_json, token_usage_json, agent_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                symbol,
                company,
                fiscal_year,
                fiscal_quarter,
                call_date,
                sector,
                exchange,
                post_return,
                prediction,
                confidence,
                1 if correct else 0 if correct is not None else None,
                json.dumps(agent_result or {}),
                json.dumps(token_usage or {}),
                agent_notes or "",
            ),
        )


def list_calls(
    symbol: Optional[str] = None,
    sector: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    ret_min: Optional[float] = None,
    ret_max: Optional[float] = None,
    prediction: Optional[str] = None,
    correct: Optional[bool] = None,
    sort: str = "date_desc",
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    init_db()
    where: List[str] = []
    params: List[Any] = []

    def add_clause(cond: str, val: Any):
        where.append(cond)
        params.append(val)

    if symbol:
        add_clause("symbol = ?", symbol.upper())
    if sector:
        add_clause("sector = ?", sector)
    if date_from:
        add_clause("call_date >= ?", date_from)
    if date_to:
        add_clause("call_date <= ?", date_to)
    if ret_min is not None:
        add_clause("post_return >= ?", ret_min)
    if ret_max is not None:
        add_clause("post_return <= ?", ret_max)
    if prediction:
        add_clause("prediction = ?", prediction.upper())
    if correct is not None:
        add_clause("correct = ?", 1 if correct else 0)

    order_by = {
        "date_desc": "call_date DESC",
        "date_asc": "call_date ASC",
        "abs_return_desc": "ABS(post_return) DESC",
        "return_desc": "post_return DESC",
        "conf_desc": "confidence DESC",
    }.get(sort, "call_date DESC")

    sql = "SELECT job_id, symbol, company, fiscal_year, fiscal_quarter, call_date, sector, exchange, post_return, prediction, confidence, correct FROM call_results"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" ORDER BY {order_by} LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with _get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_call(job_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM call_results WHERE job_id = ?", (job_id,)).fetchone()
    if not row:
        return None
    data = dict(row)
    data["agent_result"] = json.loads(data.pop("agent_result_json") or "{}")
    data["token_usage"] = json.loads(data.pop("token_usage_json") or "{}")
    return data


def get_cached_payload(symbol: str, fiscal_year: int, fiscal_quarter: int, max_age_minutes: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Return cached full analysis payload for the given symbol/year/quarter if present.
    """
    init_db()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT payload_json, created_at FROM call_cache WHERE symbol=? AND fiscal_year=? AND fiscal_quarter=?",
            (symbol.upper(), fiscal_year, fiscal_quarter),
        ).fetchone()
    if not row:
        return None
    if max_age_minutes is not None:
        try:
            cur = sqlite3.connect(":memory:")
            cur.execute("SELECT datetime('now','-%d minutes') AS cutoff" % max_age_minutes)
            cutoff = cur.fetchone()[0]
            if row["created_at"] < cutoff:
                return None
        except Exception:
            pass
    try:
        return json.loads(row["payload_json"])
    except Exception:
        return None


def set_cached_payload(symbol: str, fiscal_year: int, fiscal_quarter: int, payload: Dict[str, Any]) -> None:
    """
    Store the full analysis payload for reuse.
    """
    init_db()
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO call_cache (symbol, fiscal_year, fiscal_quarter, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (symbol.upper(), fiscal_year, fiscal_quarter, json.dumps(payload or {})),
        )


def get_fmp_cache(cache_key: str, max_age_minutes: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Fetch cached FMP payload by key. If max_age_minutes is provided, skip stale entries.
    """
    init_db()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT payload_json, created_at FROM fmp_cache WHERE cache_key=?",
            (cache_key,),
        ).fetchone()
    if not row:
        return None
    if max_age_minutes is not None:
        try:
            cur = sqlite3.connect(":memory:")
            cur.execute("SELECT datetime('now','-%d minutes') AS cutoff" % max_age_minutes)
            cutoff = cur.fetchone()[0]
            if row["created_at"] < cutoff:
                return None
        except Exception:
            pass
    try:
        return json.loads(row["payload_json"])
    except Exception:
        return None


def set_fmp_cache(cache_key: str, payload: Dict[str, Any]) -> None:
    """
    Store FMP payload by key.
    """
    init_db()
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO fmp_cache (cache_key, payload_json)
            VALUES (?, ?)
            """,
            (cache_key, json.dumps(payload or {})),
        )
