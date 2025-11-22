from datetime import datetime
from typing import Dict

from agentic_rag_bridge import AgenticRagBridgeError, run_single_call_from_context
from neo4j_ingest import Neo4jIngestError, ingest_context_into_neo4j
from fmp_client import get_earnings_context


def run_agentic_rag(context: Dict) -> Dict:
    """
    Call the real Agentic RAG pipeline via the bridge module.
    """
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
    context = get_earnings_context(symbol, year, quarter)
    agentic_result = run_agentic_rag(context)

    return {
        "symbol": symbol,
        "year": year,
        "quarter": quarter,
        "transcript_date": context.get("transcript_date"),
        "agentic_result": agentic_result,
        "context": context,
    }
