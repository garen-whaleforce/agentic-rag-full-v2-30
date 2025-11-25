import os
from datetime import datetime
from typing import Dict
from uuid import uuid4

from agentic_rag_bridge import AgenticRagBridgeError, run_single_call_from_context
from neo4j_ingest import Neo4jIngestError, ingest_context_into_neo4j, ingest_recent_history_into_neo4j
from fmp_client import get_earnings_context
from storage import record_analysis


def run_agentic_rag(context: Dict) -> Dict:
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
        ingest_recent_history_into_neo4j(context, max_quarters=history_quarters)
    except Neo4jIngestError as exc:
        _add_ingest_warning(f"Historical ingest failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        _add_ingest_warning(f"Historical ingest failed: {exc}")

    # On-the-fly Neo4j ingestion so helper agents have facts to use.
    try:
        ingest_context_into_neo4j(context)
    except Neo4jIngestError as exc:
        # Keep analyzing even if ingestion failed; surface a hint in metadata.
        context.setdefault("ingest_warning", str(exc))
    except Exception as exc:  # noqa: BLE001
        context.setdefault("ingest_warning", f"Neo4j ingestion failed: {exc}")

    try:
        result = run_single_call_from_context(context)
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


def analyze_earnings(symbol: str, year: int, quarter: int) -> Dict:
    """
    High-level orchestration: build context and run the Agentic RAG bridge.
    """
    job_id = str(uuid4())
    context = get_earnings_context(symbol, year, quarter)
    agentic_result = run_agentic_rag(context)
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
    except Exception:
        # Do not block API if persistence fails
        pass

    return {
        "symbol": symbol,
        "year": year,
        "quarter": quarter,
        "transcript_date": context.get("transcript_date"),
        "calendar_year": context.get("calendar_year"),
        "calendar_quarter": context.get("calendar_quarter"),
        "job_id": job_id,
        "agentic_result": agentic_result,
        "context": context,
    }
