"""
AWS DB-based Helper Agents
==========================
These agents use AWS PostgreSQL directly instead of Neo4j vector search.
Faster and more reliable for batch processing.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for aws_fmp_db import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from aws_fmp_db import (
    get_historical_financials,
    get_historical_transcripts,
    get_peer_financials,
    get_company_profile,
    get_earnings_surprise,
)

from agents.prompts.prompts import (
    get_financials_system_message,
    get_historical_earnings_system_message,
    get_comparative_system_message,
    financials_statement_agent_prompt,
    historical_earnings_agent_prompt,
    comparative_agent_prompt,
)
from utils.llm import build_chat_client


class TokenTracker:
    """Aggregate token usage and rough cost estimation per run."""

    def __init__(self) -> None:
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.model_used = "gpt-4o-mini"

    def add_usage(self, input_tokens: int, output_tokens: int, model: str | None = None) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        if model is not None:
            self.model_used = model

        if model:
            lowered = model.lower()
            if "gpt-4o" in lowered:
                self.total_cost_usd += (input_tokens * 0.000005) + (output_tokens * 0.000015)
            elif "gpt-4" in lowered:
                self.total_cost_usd += (input_tokens * 0.00003) + (output_tokens * 0.00006)
            elif "gpt-3.5" in lowered:
                self.total_cost_usd += (input_tokens * 0.0000015) + (output_tokens * 0.000002)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "model": self.model_used,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "cost_usd": self.total_cost_usd
        }


class AwsHistoricalPerformanceAgent:
    """
    Compare current-quarter facts with prior financial statements.
    Uses AWS PostgreSQL instead of Neo4j.
    """

    def __init__(
        self,
        credentials_file: str = "credentials.json",
        model: str = "gpt-5-mini",
        temperature: float = 1.0,
    ) -> None:
        creds = json.loads(Path(credentials_file).read_text())
        self.client, resolved_model = build_chat_client(creds, model)
        self.model = resolved_model
        self.temperature = temperature
        self.token_tracker = TokenTracker()

    def _format_financials(self, financials: Dict) -> str:
        """Format financial statements for the prompt."""
        lines = []

        for stmt in financials.get("income", []):
            date = stmt.get("date", "?")
            period = stmt.get("period", "?")
            lines.append(f"Income ({date}, {period}):")
            lines.append(f"  Revenue: {stmt.get('revenue', 'N/A'):,}" if stmt.get('revenue') else "  Revenue: N/A")
            lines.append(f"  Net Income: {stmt.get('net_income', 'N/A'):,}" if stmt.get('net_income') else "  Net Income: N/A")
            lines.append(f"  EPS: {stmt.get('eps', 'N/A')}")
            lines.append(f"  Gross Profit: {stmt.get('gross_profit', 'N/A'):,}" if stmt.get('gross_profit') else "  Gross Profit: N/A")

        for stmt in financials.get("balance", []):
            date = stmt.get("date", "?")
            lines.append(f"Balance Sheet ({date}):")
            lines.append(f"  Total Assets: {stmt.get('total_assets', 'N/A'):,}" if stmt.get('total_assets') else "  Total Assets: N/A")
            lines.append(f"  Total Debt: {stmt.get('total_debt', 'N/A'):,}" if stmt.get('total_debt') else "  Total Debt: N/A")
            lines.append(f"  Cash: {stmt.get('cash_and_cash_equivalents', 'N/A'):,}" if stmt.get('cash_and_cash_equivalents') else "  Cash: N/A")

        for stmt in financials.get("cashFlow", []):
            date = stmt.get("date", "?")
            lines.append(f"Cash Flow ({date}):")
            lines.append(f"  Operating CF: {stmt.get('operating_cash_flow', 'N/A'):,}" if stmt.get('operating_cash_flow') else "  Operating CF: N/A")
            lines.append(f"  Free CF: {stmt.get('free_cash_flow', 'N/A'):,}" if stmt.get('free_cash_flow') else "  Free CF: N/A")

        return "\n".join(lines) if lines else "No historical financial data available."

    def run(
        self,
        facts: List[Dict[str, str]],
        row: Dict,
        quarter: str,
        ticker: Optional[str] = None,
        top_n: int = 5,
    ) -> str:
        """Compare facts with historical financials from AWS DB."""
        self.token_tracker = TokenTracker()

        ticker = ticker or row.get("ticker", "")
        if not ticker:
            return None

        # Parse quarter to get date for historical lookup
        # Quarter format: "2024-Q1"
        try:
            year, q = quarter.split("-Q")
            # Approximate date for the quarter
            quarter_month = {"1": "03", "2": "06", "3": "09", "4": "12"}
            before_date = f"{year}-{quarter_month.get(q, '12')}-01"
        except:
            before_date = "2025-01-01"

        # Get historical financials from AWS DB
        historical = get_historical_financials(ticker, before_date, limit=4)

        if not historical.get("income") and not historical.get("balance") and not historical.get("cashFlow"):
            return None

        # Format for prompt
        historical_text = self._format_financials(historical)

        # Build prompt
        facts_text = "\n".join([
            f"- {f.get('metric', '?')}: {f.get('value', '?')} ({f.get('context', '')})"
            for f in facts[:20]
        ])

        prompt = f"""Analyze these current quarter facts against historical financial performance:

**Current Quarter Facts ({ticker}, {quarter}):**
{facts_text}

**Historical Financial Statements:**
{historical_text}

Compare the current facts with historical trends. Identify:
1. Significant changes from prior quarters
2. Trends in key metrics (revenue, profit, cash flow)
3. Any concerning or positive patterns

Provide a concise analysis (2-3 paragraphs)."""

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": get_financials_system_message()},
                {"role": "user", "content": prompt},
            ],
            top_p=1,
        )

        if hasattr(resp, 'usage') and resp.usage:
            self.token_tracker.add_usage(
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
                model=self.model
            )

        return resp.choices[0].message.content.strip()


class AwsHistoricalEarningsAgent:
    """
    Compare current facts with the firm's own historical earnings calls.
    Uses AWS PostgreSQL instead of Neo4j.
    """

    def __init__(
        self,
        credentials_file: str = "credentials.json",
        model: str = "gpt-5-mini",
        temperature: float = 1.0,
    ) -> None:
        creds = json.loads(Path(credentials_file).read_text())
        self.client, resolved_model = build_chat_client(creds, model)
        self.model = resolved_model
        self.temperature = temperature
        self.token_tracker = TokenTracker()

    def run(
        self,
        facts: List[Dict[str, str]],
        ticker: str,
        quarter: str,
        top_k: int = 4,
    ) -> str:
        """Compare facts with historical earnings calls from AWS DB."""
        self.token_tracker = TokenTracker()

        if not ticker:
            return None

        # Parse quarter
        try:
            year, q = quarter.split("-Q")
            year = int(year)
            q = int(q)
        except:
            return None

        # Get historical transcripts from AWS DB
        historical = get_historical_transcripts(ticker, year, q, limit=top_k)

        if not historical:
            return None

        # Extract key excerpts from historical transcripts (first 2000 chars each)
        historical_excerpts = []
        for h in historical:
            content = h.get("content", "")[:2000]
            historical_excerpts.append(
                f"**{h['year']}-Q{h['quarter']}** ({h.get('date', 'N/A')}):\n{content}..."
            )

        # Build prompt
        facts_text = "\n".join([
            f"- {f.get('metric', '?')}: {f.get('value', '?')} ({f.get('context', '')})"
            for f in facts[:15]
        ])

        prompt = f"""Compare these current quarter facts with the company's historical earnings calls:

**Current Quarter Facts ({ticker}, {quarter}):**
{facts_text}

**Historical Earnings Call Excerpts:**
{chr(10).join(historical_excerpts[:3])}

Analyze:
1. How does management's tone/message compare to previous quarters?
2. Are there recurring themes or concerns?
3. Any notable changes in guidance or outlook?

Provide a concise analysis (2-3 paragraphs)."""

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": get_historical_earnings_system_message()},
                {"role": "user", "content": prompt},
            ],
            top_p=1,
        )

        if hasattr(resp, 'usage') and resp.usage:
            self.token_tracker.add_usage(
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
                model=self.model
            )

        return resp.choices[0].message.content.strip()


class AwsComparativeAgent:
    """
    Compare facts against peer companies in the same sector.
    Uses AWS PostgreSQL instead of Neo4j.
    """

    def __init__(
        self,
        credentials_file: str = "credentials.json",
        model: str = "gpt-5-mini",
        sector_map: dict = None,
        temperature: float = 1.0,
        **kwargs,
    ) -> None:
        creds = json.loads(Path(credentials_file).read_text())
        self.client, resolved_model = build_chat_client(creds, model)
        self.model = resolved_model
        self.temperature = temperature
        self.token_tracker = TokenTracker()
        self.sector_map = sector_map or {}

    def run(
        self,
        facts: List[Dict[str, str]],
        ticker: str,
        quarter: str,
        peers: list[str] | None = None,
        sector: str | None = None,
        top_k: int = 5,
    ) -> str:
        """Compare facts with peer companies from AWS DB."""
        self.token_tracker = TokenTracker()

        if not ticker:
            return None

        # Get sector if not provided
        if not sector:
            profile = get_company_profile(ticker)
            sector = profile.get("sector") if profile else None

        if not sector:
            return None

        # Parse quarter for date
        try:
            year, q = quarter.split("-Q")
            quarter_month = {"1": "03", "2": "06", "3": "09", "4": "12"}
            as_of_date = f"{year}-{quarter_month.get(q, '12')}-01"
        except:
            as_of_date = "2025-01-01"

        # Get peer financials from AWS DB
        peer_data = get_peer_financials(sector, ticker, as_of_date, limit=top_k)

        if not peer_data:
            return None

        # Format peer data
        peer_lines = []
        for p in peer_data:
            fin = p.get("financials") or {}
            peer_lines.append(
                f"**{p['symbol']}** ({p.get('name', 'N/A')}):\n"
                f"  Revenue: {fin.get('revenue', 'N/A'):,}" if fin.get('revenue') else f"  Revenue: N/A"
            )
            if fin.get('net_income'):
                peer_lines[-1] += f", Net Income: {fin['net_income']:,}"
            if fin.get('eps'):
                peer_lines[-1] += f", EPS: {fin['eps']}"

        # Build prompt
        facts_text = "\n".join([
            f"- {f.get('metric', '?')}: {f.get('value', '?')} ({f.get('context', '')})"
            for f in facts[:15]
        ])

        prompt = f"""Compare {ticker}'s performance with sector peers ({sector}):

**{ticker} Current Quarter Facts ({quarter}):**
{facts_text}

**Peer Companies ({sector} sector):**
{chr(10).join(peer_lines)}

Analyze:
1. How does {ticker} compare to peers on key metrics?
2. Is {ticker} outperforming or underperforming the sector?
3. Any notable competitive advantages or disadvantages?

Provide a concise analysis (2-3 paragraphs)."""

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": get_comparative_system_message()},
                {"role": "user", "content": prompt},
            ],
            top_p=1,
        )

        if hasattr(resp, 'usage') and resp.usage:
            self.token_tracker.add_usage(
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
                model=self.model
            )

        return resp.choices[0].message.content.strip()

    def close(self) -> None:
        """No-op for compatibility."""
        pass
