from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional


class AgenticRagBridgeError(RuntimeError):
    """Custom error for Agentic RAG bridge failures."""


REPO_NAME = "EarningsCallAgenticRag"


def _resolve_repo_path() -> Path:
    """Locate the external repo; raise with actionable guidance if missing."""
    base = Path(__file__).resolve().parent
    env_path = os.getenv("EARNINGS_RAG_PATH")
    repo_path = Path(env_path) if env_path else base / REPO_NAME
    if not repo_path.exists():
        raise AgenticRagBridgeError(
            f"找不到外部研究庫資料夾：{repo_path}. "
            "請先執行 `git clone https://github.com/la9806958/EarningsCallAgenticRag.git EarningsCallAgenticRag` "
            "並確認與本專案並排。"
        )
    return repo_path


def _ensure_sys_path(repo_path: Path) -> None:
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _env_credentials() -> Optional[Dict[str, Any]]:
    openai_key = os.getenv("OPENAI_API_KEY")
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_db = os.getenv("NEO4J_DATABASE") or "neo4j"
    if not all([openai_key, neo4j_uri, neo4j_username, neo4j_password]):
        return None
    return {
        "openai_api_key": openai_key,
        "neo4j_uri": neo4j_uri,
        "neo4j_username": neo4j_username,
        "neo4j_password": neo4j_password,
        "neo4j_database": neo4j_db,
    }


def _credentials_path(repo_path: Path) -> Path:
    cred = repo_path / "credentials.json"
    if not cred.exists():
        env_creds = _env_credentials()
        if env_creds:
            try:
                # Avoid race: create only if missing
                cred.write_text(json.dumps(env_creds, indent=2))
            except FileExistsError:
                # Another process wrote it; keep going
                pass
        else:
            raise AgenticRagBridgeError(
                f"外部庫的 credentials.json 未找到：{cred}. "
                "請依照 EarningsCallAgenticRag README 填入 OpenAI 與 Neo4j 設定，或在環境變數提供 OPENAI_API_KEY / NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD。"
            )
    return cred


def _load_sector_map(repo_path: Path) -> Dict[str, str]:
    """Best-effort load GICS sector map if存在."""
    candidates = [
        repo_path / "gics_sector_map_nyse.csv",
        repo_path / "gics_sector_map_nasdaq.csv",
    ]
    import pandas as pd  # Lazy import; included in requirements

    for csv_path in candidates:
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                cols = {c.lower(): c for c in df.columns}
                ticker_col = cols.get("ticker") or cols.get("symbol")
                sector_col = cols.get("sector") or cols.get("gics_sector")
                if ticker_col and sector_col:
                    return {
                        str(t).upper(): str(s)
                        for t, s in zip(df[ticker_col], df[sector_col])
                        if pd.notna(t) and pd.notna(s)
                    }
            except Exception:
                continue
    return {}


def _summarize_financials(financials: Optional[Dict[str, Any]]) -> str:
    """Create a compact string for the main agent prompt."""
    if not financials:
        return "No structured financials supplied."

    parts: List[str] = []
    income = financials.get("income") or []
    balance = financials.get("balance") or []
    cash = financials.get("cashFlow") or []

    def _line(label: str, rows: List[dict], keys: List[str]) -> Optional[str]:
        if not rows:
            return None
        latest = rows[0] if isinstance(rows[0], dict) else {}
        date = (
            latest.get("date")
            or latest.get("calendarYear")
            or latest.get("fillingDate")
            or latest.get("period")
        )
        metrics = []
        for k in keys:
            if k in latest and latest[k] not in (None, ""):
                metrics.append(f"{k}={latest[k]}")
        if not metrics:
            metrics.append("no key metrics detected")
        return f"{label} [{date or 'n/a'}]: " + ", ".join(metrics)

    income_line = _line("Income", income, ["revenue", "netIncome", "eps", "grossProfit"])
    balance_line = _line("Balance", balance, ["totalAssets", "totalLiabilities", "cashAndCashEquivalents"])
    cash_line = _line("CashFlow", cash, ["operatingCashFlow", "freeCashFlow"])

    for ln in (income_line, balance_line, cash_line):
        if ln:
            parts.append(ln)

    return "\n".join(parts) if parts else "Financial statements present but could not summarize."


@contextmanager
def _push_dir(path: Path):
    cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(cwd)


def run_single_call_from_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the real Agentic RAG pipeline from the external repo for a single earnings call.

    Returns a result dict with at least: prediction, confidence, summary, reasons, raw.
    """
    repo_path = _resolve_repo_path()
    _ensure_sys_path(repo_path)
    cred_path = _credentials_path(repo_path)

    try:
        from agents.mainAgent import MainAgent
        from agents.comparativeAgent import ComparativeAgent
        from agents.historicalPerformanceAgent import HistoricalPerformanceAgent
        from agents.historicalEarningsAgent import HistoricalEarningsAgent
    except Exception as exc:  # noqa: BLE001
        raise AgenticRagBridgeError(f"匯入 Agentic RAG 模組失敗：{exc}") from exc

    symbol = (context.get("symbol") or context.get("ticker") or "").upper()
    year = context.get("year")
    quarter = context.get("quarter")
    transcript_text = context.get("transcript_text") or ""

    if not symbol or not year or not quarter:
        raise AgenticRagBridgeError("context 缺少必填欄位：symbol、year、quarter。")

    quarter_label = f"{year}-Q{quarter}"
    sector_map = _load_sector_map(repo_path)

    with _push_dir(repo_path):
        comparative_agent = ComparativeAgent(credentials_file=str(cred_path), sector_map=sector_map or None)
        financials_agent = HistoricalPerformanceAgent(credentials_file=str(cred_path))
        past_calls_agent = HistoricalEarningsAgent(credentials_file=str(cred_path))
        main_agent = MainAgent(
            credentials_file=str(cred_path),
            comparative_agent=comparative_agent,
            financials_agent=financials_agent,
            past_calls_agent=past_calls_agent,
        )

        # Extract and annotate facts from transcript
        facts = main_agent.extract(transcript_text)
        for f in facts:
            f.setdefault("ticker", symbol)
            f.setdefault("quarter", quarter_label)

        row = {
            "ticker": symbol,
            "q": quarter_label,
            "transcript": transcript_text,
            "sector": context.get("sector"),
        }
        financials_text = _summarize_financials(context.get("financials"))
        agent_output = main_agent.run(facts, row, mem_txt=None, original_transcript=transcript_text, financial_statements_facts=financials_text)

    if not isinstance(agent_output, dict):
        agent_output = {"raw_output": agent_output}

    def _infer_direction(summary: Optional[str]) -> tuple[str, Optional[float]]:
        if not summary:
            return "UNKNOWN", None
        import re

        match = re.search(r"Direction:\s*(\d+)", summary)
        if match:
            score = int(match.group(1))
            if score >= 7:
                return "UP", score / 10
            if score <= 3:
                return "DOWN", score / 10
            return "NEUTRAL", score / 10
        lowered = summary.lower()
        if any(k in lowered for k in ["up", "increase", "growth", "record", "beat"]):
            return "UP", 0.6
        if any(k in lowered for k in ["down", "decline", "miss", "pressure", "headwind"]):
            return "DOWN", 0.4
        return "UNKNOWN", None

    notes = agent_output.get("notes") or {}

    def _keep(val: Optional[str]) -> Optional[str]:
        if not val:
            return None
        normalized = str(val).strip()
        if normalized.lower() in {"n/a", "na", "none"}:
            return None
        return normalized

    reasons = [
        f"financials: {notes.get('financials')}" if _keep(notes.get("financials")) else None,
        f"past calls: {notes.get('past')}" if _keep(notes.get("past")) else None,
        f"peers: {notes.get('peers')}" if _keep(notes.get("peers")) else None,
    ]
    reasons = [r for r in reasons if r]

    if not reasons:
        # Fallback：取前 3 條提取的事實做理由摘要
        top_facts = facts[:3]
        for f in top_facts:
            metric = f.get("metric") or "metric"
            val = f.get("value") or ""
            ctx = f.get("context") or f.get("reason") or ""
            reasons.append(f"{metric}: {val} {ctx}".strip())

    prediction, confidence = _infer_direction(agent_output.get("summary"))

    return {
        "prediction": prediction,
        "confidence": confidence,
        "summary": agent_output.get("summary"),
        "reasons": reasons,
        "raw": agent_output,
    }
