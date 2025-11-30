import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Optional
from uuid import uuid4

from agentic_rag_bridge import AgenticRagBridgeError, run_single_call_from_context
from neo4j_ingest import Neo4jIngestError, ingest_context_into_neo4j, ingest_recent_history_into_neo4j
from fmp_client import get_earnings_context, get_earnings_context_async
from storage import (
    record_analysis,
    get_cached_payload,
    set_cached_payload,
)
from redis_cache import cache_get_json, cache_set_json
from earnings_backtest import compute_earnings_backtest

logger = logging.getLogger(__name__)
PAYLOAD_CACHE_MINUTES = int(os.getenv("PAYLOAD_CACHE_MINUTES", "1440"))  # DB cache validity in minutes (default 1 day)
REDIS_PAYLOAD_TTL_SECONDS = int(os.getenv("REDIS_PAYLOAD_TTL_SECONDS", "3600"))  # Redis TTL in seconds (default 1 hour)


def run_agentic_rag(
    context: Dict,
    main_model: Optional[str] = None,
    helper_model: Optional[str] = None,
) -> Dict:
    """
    Call the real Agentic RAG pipeline via the bridge module.
    """
    def _add_ingest_warning(msg: str) -> None:
        if not msg:
            return
        if context.get("ingest_warning"):
            context["ingest_warning"] = f"{context['ingest_warning']} | {msg}"
        else:
            context["ingest_warning"] = msg

    # First, backfill recent historical quarters so helper agents have past facts.
    history_quarters = 4
    try:
        history_quarters = int(os.getenv("INGEST_HISTORY_QUARTERS", "4"))
    except Exception:
        history_quarters = 4

    try:
        _retry(lambda: ingest_recent_history_into_neo4j(context, max_quarters=history_quarters))
    except Neo4jIngestError as exc:
        logger.warning("Historical ingest failed: %s", exc)
        _add_ingest_warning(f"Historical ingest failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Historical ingest failed: %s", exc)
        _add_ingest_warning(f"Historical ingest failed: {exc}")

    # On-the-fly Neo4j ingestion so helper agents have facts to use.
    try:
        _retry(lambda: ingest_context_into_neo4j(context))
    except Neo4jIngestError as exc:
        # Keep analyzing even if ingestion failed; surface a hint in metadata.
        context.setdefault("ingest_warning", str(exc))
        logger.warning("Neo4j ingestion failed: %s", exc)
    except Exception as exc:  # noqa: BLE001
        context.setdefault("ingest_warning", f"Neo4j ingestion failed: {exc}")
        logger.warning("Neo4j ingestion failed: %s", exc)

    try:
        result = run_single_call_from_context(
            context,
            main_model=main_model,
            helper_model=helper_model,
        )
    except AgenticRagBridgeError:
        # Propagate bridge-specific errors directly for clearer API feedback
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Agentic RAG pipeline failure: {exc}") from exc

    if not isinstance(result, dict):
        result = {"raw_output": result}

    metadata = result.setdefault("metadata", {})
    metadata.setdefault("generated_at", datetime.utcnow().isoformat() + "Z")
    metadata.setdefault("engine", "EarningsCallAgenticRag")
    metadata.setdefault("transcript_excerpt", (context.get("transcript_text") or "")[:280])
    if context.get("ingest_warning"):
        metadata.setdefault("ingest_warning", context["ingest_warning"])
    return result


def analyze_earnings(
    symbol: str,
    year: int,
    quarter: int,
    main_model: Optional[str] = None,
    helper_model: Optional[str] = None,
) -> Dict:
    """
    High-level orchestration: build context and run the Agentic RAG bridge.
    """
    job_id = str(uuid4())
    context = get_earnings_context(symbol, year, quarter)
    agentic_result = run_agentic_rag(
        context,
        main_model=main_model,
        helper_model=helper_model,
    )
    if not isinstance(agentic_result, dict):
        agentic_result = {"raw_output": agentic_result}

    # Persist summary for listing/detail pages
    try:
        raw = agentic_result.get("raw") if isinstance(agentic_result, dict) else {}
        token_usage = raw.get("token_usage") if isinstance(raw, dict) else None
        notes = raw.get("notes") if isinstance(raw, dict) else None
        # Determine correctness if post return is available
        post_ret = context.get("post_earnings_return")
        pred = agentic_result.get("prediction") if isinstance(agentic_result, dict) else None
        correct = None
        if post_ret is not None and pred:
            pred_upper = str(pred).upper()
            if pred_upper == "UP":
                correct = post_ret > 0
            elif pred_upper == "DOWN":
                correct = post_ret < 0
            elif pred_upper == "NEUTRAL":
                correct = abs(post_ret) < 0.01

        record_analysis(
            job_id=job_id,
            symbol=symbol,
            fiscal_year=year,
            fiscal_quarter=quarter,
            call_date=context.get("transcript_date"),
            sector=context.get("sector"),
            exchange=context.get("exchange"),
            post_return=post_ret,
            prediction=pred,
            confidence=agentic_result.get("confidence") if isinstance(agentic_result, dict) else None,
            correct=correct,
            agent_result=agentic_result if isinstance(agentic_result, dict) else {},
            token_usage=token_usage,
            agent_notes=str(notes) if notes else None,
            company=context.get("company"),
        )
    except Exception as exc:
        # Do not block API if persistence fails
        logger.exception("record_analysis failed", exc_info=exc)

    # Compute earnings backtest (BMO/AMC price change)
    backtest = None
    try:
        backtest = compute_earnings_backtest(
            symbol,
            context.get("transcript_date") or "",
            context.get("transcript_text") or "",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("compute_earnings_backtest failed for %s: %s", symbol, exc)

    payload = {
        "symbol": symbol,
        "year": year,
        "quarter": quarter,
        "transcript_date": context.get("transcript_date"),
        "calendar_year": context.get("calendar_year"),
        "calendar_quarter": context.get("calendar_quarter"),
        "post_return_meta": context.get("post_return_meta"),
        "post_earnings_return": context.get("post_earnings_return"),
        "job_id": job_id,
        "agentic_result": agentic_result,
        "context": context,
        "backtest": backtest,
    }

    return payload


async def analyze_earnings_async(
    symbol: str,
    year: int,
    quarter: int,
    main_model: Optional[str] = None,
    helper_model: Optional[str] = None,
    skip_cache: bool = False,
) -> Dict:
    """
    Async wrapper: fetch context in parallel and run agentic pipeline in thread to avoid blocking event loop.
    """
    cache_key = f"call:{symbol.upper()}:{year}:Q{quarter}"

    if not skip_cache:
        # 1) Redis cache
        cached_payload = await cache_get_json(cache_key)
        if isinstance(cached_payload, dict) and cached_payload.get("symbol"):
            return cached_payload

        # 2) DB cache
        try:
            db_cached = get_cached_payload(
                symbol=symbol,
                fiscal_year=year,
                fiscal_quarter=quarter,
                max_age_minutes=PAYLOAD_CACHE_MINUTES,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("get_cached_payload failed, ignoring cache", exc_info=exc)
            db_cached = None

        if isinstance(db_cached, dict) and db_cached.get("symbol"):
            try:
                await cache_set_json(cache_key, db_cached, REDIS_PAYLOAD_TTL_SECONDS)
            except Exception:
                pass
            return db_cached

    job_id = str(uuid4())
    context = await get_earnings_context_async(symbol, year, quarter)
    agentic_result = await asyncio.to_thread(
        run_agentic_rag,
        context,
        main_model,
        helper_model,
    )
    if not isinstance(agentic_result, dict):
        agentic_result = {"raw_output": agentic_result}

    try:
        raw = agentic_result.get("raw") if isinstance(agentic_result, dict) else {}
        token_usage = raw.get("token_usage") if isinstance(raw, dict) else None
        notes = raw.get("notes") if isinstance(raw, dict) else None
        post_ret = context.get("post_earnings_return")
        pred = agentic_result.get("prediction") if isinstance(agentic_result, dict) else None
        correct = None
        if post_ret is not None and pred:
            pred_upper = str(pred).upper()
            if pred_upper == "UP":
                correct = post_ret > 0
            elif pred_upper == "DOWN":
                correct = post_ret < 0
            elif pred_upper == "NEUTRAL":
                correct = abs(post_ret) < 0.01

        await asyncio.to_thread(
            record_analysis,
            job_id,
            symbol,
            year,
            quarter,
            context.get("transcript_date"),
            context.get("sector"),
            context.get("exchange"),
            post_ret,
            pred,
            agentic_result.get("confidence") if isinstance(agentic_result, dict) else None,
            correct,
            agentic_result if isinstance(agentic_result, dict) else {},
            token_usage,
            str(notes) if notes else None,
            context.get("company"),
        )
    except Exception as exc:
        logger.exception("record_analysis failed", exc_info=exc)

    # Compute earnings backtest (BMO/AMC price change)
    backtest = None
    try:
        backtest = await asyncio.to_thread(
            compute_earnings_backtest,
            symbol,
            context.get("transcript_date") or "",
            context.get("transcript_text") or "",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("compute_earnings_backtest failed for %s: %s", symbol, exc)

    payload = {
        "symbol": symbol,
        "year": year,
        "quarter": quarter,
        "transcript_date": context.get("transcript_date"),
        "calendar_year": context.get("calendar_year"),
        "calendar_quarter": context.get("calendar_quarter"),
        "post_return_meta": context.get("post_return_meta"),
        "post_earnings_return": context.get("post_earnings_return"),
        "job_id": job_id,
        "agentic_result": agentic_result,
        "context": context,
        "backtest": backtest,
    }

    try:
        set_cached_payload(symbol, year, quarter, payload)
    except Exception as exc:  # noqa: BLE001
        logger.exception("set_cached_payload failed, ignoring", exc_info=exc)

    try:
        await cache_set_json(cache_key, payload, REDIS_PAYLOAD_TTL_SECONDS)
    except Exception:
        pass

    return payload
# Simple retry helper for Neo4j ingest
def _retry(func, attempts: int = 3, delay: float = 1.0):
    import time

    last_exc = None
    for _ in range(attempts):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(delay)
    if last_exc:
        raise last_exc
