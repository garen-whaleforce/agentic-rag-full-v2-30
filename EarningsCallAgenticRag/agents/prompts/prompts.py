"""Prompt utilities for the Agentic RAG earnings-call workflow."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from prompt_service import get_prompt_override

__all__ = [
    "get_main_agent_system_message",
    "get_extraction_system_message",
    "get_delegation_system_message",
    "get_comparative_system_message",
    "get_historical_earnings_system_message",
    "get_financials_system_message",
    "comparative_agent_prompt",
    "historical_earnings_agent_prompt",
    "main_agent_prompt",
    "facts_extraction_prompt",
    "facts_delegation_prompt",
    "peer_discovery_ticker_prompt",
    "financials_statement_agent_prompt",
    "memory",
    "baseline_prompt",
]

# === Main Agent System Message ===
_DEFAULT_MAIN_AGENT_SYSTEM_MESSAGE = """
You are a seasoned portfolio manager.
""".strip()


def get_main_agent_system_message() -> str:
    return get_prompt_override("MAIN_AGENT_SYSTEM_MESSAGE", _DEFAULT_MAIN_AGENT_SYSTEM_MESSAGE)


# === Extraction System Message ===
_DEFAULT_EXTRACTION_SYSTEM_MESSAGE = """
You are a precise extraction bot.
""".strip()


def get_extraction_system_message() -> str:
    return get_prompt_override("EXTRACTION_SYSTEM_MESSAGE", _DEFAULT_EXTRACTION_SYSTEM_MESSAGE)


# === Delegation System Message ===
_DEFAULT_DELEGATION_SYSTEM_MESSAGE = """
Route each fact.
""".strip()


def get_delegation_system_message() -> str:
    return get_prompt_override("DELEGATION_SYSTEM_MESSAGE", _DEFAULT_DELEGATION_SYSTEM_MESSAGE)


# === Comparative System Message ===
_DEFAULT_COMPARATIVE_SYSTEM_MESSAGE = "You are a financial forecasting assistant."


def get_comparative_system_message() -> str:
    return get_prompt_override("COMPARATIVE_SYSTEM_MESSAGE", _DEFAULT_COMPARATIVE_SYSTEM_MESSAGE)


# === Historical Earnings System Message ===
_DEFAULT_HISTORICAL_EARNINGS_SYSTEM_MESSAGE = "You are a financial forecasting assistant."


def get_historical_earnings_system_message() -> str:
    return get_prompt_override("HISTORICAL_EARNINGS_SYSTEM_MESSAGE", _DEFAULT_HISTORICAL_EARNINGS_SYSTEM_MESSAGE)


# === Financials System Message ===
_DEFAULT_FINANCIALS_SYSTEM_MESSAGE = "You are a financial forecasting assistant."


def get_financials_system_message() -> str:
    return get_prompt_override("FINANCIALS_SYSTEM_MESSAGE", _DEFAULT_FINANCIALS_SYSTEM_MESSAGE)


# ============================================================================
# PROMPT FUNCTIONS
# ============================================================================

def comparative_agent_prompt(
    facts: List[Dict[str, Any]],
    related_facts: List[Dict[str, Any]],
    self_ticker: str | None = None,
) -> str:
    """Return the prompt for the *Comparative Peers* analysis agent.

    Parameters
    ----------
    facts
        A list of facts extracted from the current firm's earnings call.
    related_facts
        A list of facts from comparable peer firms.
    self_ticker
        The ticker symbol of the firm being analyzed.
    """
    ticker_section = f"\n\nThe ticker of the firm being analyzed is: {self_ticker}" if self_ticker else ""
    return f"""
You are analyzing a company's earnings call transcript alongside statements made by similar firms.{ticker_section}

The batch of facts about the firm is:
{json.dumps(facts, indent=2)}

Comparable firms discuss the facts in the following way:
{json.dumps(related_facts, indent=2)}

Your task is:
- Describe how the firm's reasoning about their own performance differs from other firms, for each fact if possible.
- Cite factual evidence from historical calls.

Keep your analysis concise. Do not discuss areas not mentioned.

Bilingual requirement:
- First, produce the full analysis in **English**, following the exact format specified above.
- Then, produce a second section titled **"Traditional Chinese (繁體中文)"**
  where you translate the same headings, bullet points, and conclusions
  into Traditional Chinese, preserving the same ordering, structure,
  and all numeric values.
""".strip()


def historical_earnings_agent_prompt(
    fact: Dict[str, Any],
    related_facts: List[Dict[str, Any]],
    current_quarter: str | None = None,
) -> str:
    """
    Return the prompt for the *Historical Earnings* analysis agent.

    Parameters
    ----------
    fact : dict
        The current fact from the firm's latest earnings call.
    related_facts : list of dict
        A list of related facts drawn from the firm's own previous calls.
    current_quarter : str
        The current fiscal quarter (e.g., 'Q2 2025').
    """
    quarter_label = current_quarter if current_quarter else "the current quarter"
    return f"""
You are analyzing a company's earnings call transcript alongside facts from its own past earnings calls.

The list of current facts are:
{json.dumps(fact, indent=2)}

It is reported in the quarter {quarter_label}

Here is a JSON list of related facts from the firm's previous earnings calls:
{json.dumps(related_facts, indent=2)}

TASK
────
1. **Validate past guidanced**
   ▸ For every forward-looking statement made in previous quarters, state whether the firm met, beat, or missed that guidance in `{quarter_label}`.
   ▸ Reference concrete numbers (e.g., "Revenue growth was 12 % vs. the 10 % guided in 2024-Q3").
   ▸ Omit if you cannot provide a direct comparison

2. **Compare results discussed**
    ▸ Compare the results being discussed.
    ▸ Reference concrete numbers

3. **Provide supporting evidence.**
   ▸ Quote or paraphrase the relevant historical statement, then cite the matching current-quarter metric.
   ▸ Format each evidence line as
     `• <metric>: <historical statement> → <current result>`.

Keep your analysis concise. Prioritize more recent quarters. Do not discuss areas not mentioned.

Bilingual requirement:
- After you complete the analysis in English using the structure described above,
  add a second section titled **"Traditional Chinese (繁體中文)"**.
- In that section, translate each heading, bullet point, and conclusion
  into Traditional Chinese, keeping the structure and numbers identical.
""".strip()


def financials_statement_agent_prompt(
    fact: Dict[str, Any],
    similar_facts: list,
    quarter: str | None = None,
) -> str:
    """Prompt template for analysing the current fact in the context of most similar past facts."""
    quarter_label = quarter if quarter else "the current quarter"
    return f"""
You are reviewing the company's {quarter_label} earnings-call transcript and comparing a key fact to the most similar historical facts from previous quarters.

────────────────────────────────────────
Current fact (from {quarter_label}):
{json.dumps(fact, indent=2)}

Most similar past facts (from previous quarters):
{json.dumps(similar_facts, indent=2)}
────────────────────────────────────────

Your tasks:

1. **Direct comparison**
   • Compare the current fact to each of the most similar past facts. For each, note the quarter, the metric, and the value.
   • Highlight similarities, differences, and any notable trends or changes.
   • If the current value is higher/lower/similar to the most recent similar fact, state this explicitly.

2. **Supported outcomes**
   • Identify areas where management explicitly addressed historical comparisons and the numbers confirm their comments.

Focus on improvements on bottom line performance (eg. net income)

*Note: Figures may be stated in ten-thousands (万) or hundreds of millions (亿). Make sure to account for these scale differences when comparing values.*

Keep your analysis concise. Prioritize more recent quarters. Do not discuss areas not mentioned.

Bilingual requirement:
- First, output the full financial analysis in **English** exactly as specified above.
- Then, output a second section titled **"Traditional Chinese (繁體中文)"**,
  which is a translation of the English section into Traditional Chinese.
- The headings, bullet ordering, and all numeric values must be preserved.
""".strip()


def memory(all_notes, actual_return):
    """
    Combine prior note and realized move for calibration.

    Parameters
    ----------
    all_notes
        Prior research notes about the company.
    actual_return
        Realized next-day return after the prior call (string).
    """
    return f"""
    You have memory on how your previous prediction on the firm faired.
    Your previous research note is given as:
    {all_notes},
    The actual return achieved by your previous note was : {actual_return}
    """


def main_agent_prompt(
    notes,
    all_notes=None,
    original_transcript: str | None = None,
    memory_txt: str | None = None,
    financial_statements_facts: str | None = None,
    qoq_section: str | None = None,
) -> str:
    """Prompt for the *Main* decision-making agent, requesting just an
    Up/Down call plus a confidence score (0-100)."""
    transcript_section = f"\nORIGINAL EARNINGS CALL TRANSCRIPT:\n---\n{original_transcript}\n---\n" if original_transcript else ""

    financial_statements_section = ""
    if financial_statements_facts:
        financial_statements_section = f"""
---
Financial Statements Facts (YoY):
{financial_statements_facts}
---
"""

    qoq_section_str = ""
    if qoq_section:
        qoq_section_str = f"\n---\nQuarter-on-Quarter Changes:\n{qoq_section}\n---\n"

    memory_section = ""
    if memory_txt:
        memory_section = f"\n{memory_txt}\n"

    return f"""
You are a portfolio manager and you are reading an earnings call transcript.{transcript_section}
decide whether the stock price is likely to **increase ("Up") or decrease ("Down")**
one trading day after the earnings call, and assign a **Direction score** from 0 to 10.

The original transcript is:

{original_transcript}

{financial_statements_section}
+{qoq_section_str}
---
Financials-vs-History note:
{notes['financials']}

Historical-Calls note:
{notes['past']}

Peer-Comparison note:
{notes['peers']}


{memory_section}

---

Instructions:
1. Assign a confidence score (0 = strong conviction of decline, 5 = neutral, 10 = strong conviction of rise).
2. Evaluate all three notes together
3. Consider the financial statements facts when available
4. Pay special attention to the year on year changes section, especially on bottom line figures (eg. net profit)

Respond in **exactly** this format:

<Couple of sentences of Explanation>

**Summary: <Two sentences supporting your verdict with facts and evidence>, Direction : <0-10>**

Bilingual requirement:
- First, produce the full output in **English** exactly following the format
  and examples specified above, including the final "Direction" line.
- After you have produced the English output, append a second section where you
  translate the entire explanation (but not the numeric score) into
  Traditional Chinese (繁體中文), preserving the same reasoning and structure.
- Do NOT modify the English output or its format; the Chinese section must come
  AFTER the English output.
""".strip()


def facts_extraction_prompt(transcript_chunk: str) -> str:
    """
    Build the LLM prompt that asks for five specific data classes
    (Result, Forward-Looking, Risk Disclosure, Sentiment, and Macro)
    from a single earnings-call transcript chunk.
    """
    return f"""
You are a senior equity-research analyst.

### TASK
Extract **only** the following five classes from the transcript below.
Ignore moderator chatter, safe-harbor boiler-plate, and anything that doesn't match one of these classes.

1. **Result** – already-achieved financial or operating results
2. **Forward-Looking** – any explicit future projection, target, plan, or guidance
3. **Risk Disclosure** – statements highlighting current or expected obstacles
   (e.g., FX headwinds, supply-chain issues, regulation)
4. **Sentiment** – management's overall tone (Positive, Neutral, or Negative);
   cite key wording that informs your judgment.
5. **Macro** – discussion of how the macro-economic landscape affects the firm

The transcript is {transcript_chunk}

Output as many items as you can find, ideally 30-70. You MUST output more than 30 facts.
Do not include [ORG] in your response.
---

### OUTPUT RULES
* Use the exact markdown block below for **every** extracted item.
* Increment the item number sequentially (1, 2, 3 …).
* One metric per block; never combine multiple metrics.

Example output:
### Fact No. 1
- **Type:** <Result | Forward-Looking | Risk Disclosure | Sentiment | Macro>
- **Metric:** Revenue
- **Value:** "3 million dollars"
- **Reason:** Quarter was up on a daily organic basis, driven primarily by core non-pandemic product sales.

"""


def facts_delegation_prompt(facts: List) -> str:
    """Return the prompt used for routing facts to helper tools.

    Parameters
    ----------
    facts
        A list of extracted facts from the earnings call.
    """
    return f""" You are the RAG-orchestration analyst for an earnings-call workflow.

## Objective
For **each fact** listed below, decide **which (if any) of the three tools** will
help you gauge its potential impact on the company's share price **one trading
day after the call**.

### Available Tools
1. **InspectPastStatements**
   • Retrieves historical income-statement, balance-sheet, and cash-flow data
   • **Use when** the fact cites a standard, repeatable line-item
     (e.g., revenue, EBITDA, free cash flow, margins)

2. **QueryPastCalls**
   • Fetches the same metric or statement from prior earnings calls
   • **Use when** comparing management's current commentary with its own
     previous statements adds context

3. **CompareWithPeers**
   • Provides the same metric from peer companies' calls or filings
   • **Use when** competitive benchmarking clarifies whether the fact signals
     outperformance, underperformance, or parity

---
The facts are: {facts}

Output your answers in the following form:

InspectPastStatements: Fact No <2, 4, 6>
CompareWithPeers:  Fact No <10>
QueryPastCalls: Fact No <1, 3, 5>

*One fact may appear under multiple tools if multiple comparisons are helpful.*

"""

peer_discovery_ticker_prompt = """
You are a financial analyst. Based on the company with ticker {ticker}, list 5 close peer companies that are in the same or closely related industries.

Only output a Python-style list of tickers, like:
["AAPL", "GOOGL", "AMZN", "MSFT", "ORCL"]
"""


# ============================================================================
# BASELINE PROMPTS
# ============================================================================

def baseline_prompt(transcript) -> str:
    return f"""
You are a portfolio manager and you are reading an earnings call transcript.
decide whether the stock price is likely to **increase ("Up") or decrease ("Down")**
one trading day after the earnings call, and assign a **Direction score** from 0 to 10.

---
{transcript}

Instructions:
1. Assign a confidence score (0 = strong conviction of decline, 5 = neutral, 10 = strong conviction of rise).

Respond in **exactly** this format:

<Couple of sentences of Explanation>
Direction : <0-10>

""".strip()
