"""Prompt utilities for the Agentic RAG earnings-call workflow."""
from __future__ import annotations

import json
from typing import Any, Dict, List

__all__ = [
    "MAIN_AGENT_SYSTEM_MESSAGE",
    "EXTRACTION_SYSTEM_MESSAGE",
    "DELEGATION_SYSTEM_MESSAGE",
    "COMPARATIVE_SYSTEM_MESSAGE",
    "HISTORICAL_EARNINGS_SYSTEM_MESSAGE",
    "FINANCIALS_SYSTEM_MESSAGE",
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

MAIN_AGENT_SYSTEM_MESSAGE = (
    "You are a long-only portfolio manager focused on the one-trading-day "
    "price move after an earnings call."
)

EXTRACTION_SYSTEM_MESSAGE = (
    "You are a senior equity-research analyst extracting structured facts "
    "from earnings calls."
)

DELEGATION_SYSTEM_MESSAGE = (
    "You are the orchestration analyst deciding which tools should enrich "
    "each fact for trading decisions."
)

COMPARATIVE_SYSTEM_MESSAGE = (
    "You are an equity analyst specialising in cross-company comparisons "
    "within the same industry."
)

HISTORICAL_EARNINGS_SYSTEM_MESSAGE = (
    "You are an equity analyst specialising in comparing management commentary "
    "across quarters."
)

FINANCIALS_SYSTEM_MESSAGE = (
    "You are an equity analyst specialising in interpreting changes in "
    "financial statements over time."
)


def comparative_agent_prompt(
    facts: List[Dict[str, Any]],
    related_facts: List[Dict[str, Any]],
    self_ticker: str | None = None,
) -> str:
    """Prompt template for the peer-comparison helper agent."""
    ticker_section = (
        f"\nThe company you are analysing has ticker: {self_ticker}" if self_ticker else ""
    )

    return f"""
You are an equity analyst specialising in **peer comparison within the same industry**.{ticker_section}

Your goal is to evaluate how this company's latest earnings message differs from
its peers in a way that could matter for the **one-trading-day price reaction**.

### Current company facts (latest quarter)
{json.dumps(facts, indent=2, ensure_ascii=False)}

### Peer facts (similar metrics from comparable firms)
{json.dumps(related_facts, indent=2, ensure_ascii=False)}

### Tasks
1. For each important metric (revenue growth, margins, cash flow, guidance, etc.):
   - State whether the company is **stronger, weaker, or in line** vs peers.
   - Note whether management tone is **more optimistic, neutral, or cautious** vs peers.
2. Highlight **where the company is clearly differentiated** (positively or negatively).
3. End with 3 bullet points under "Implications for near-term price reaction".

### Output structure

**Relative performance vs peers**
- <Metric 1>: <stronger / weaker / in line>, <one sentence explanation>
- <Metric 2>: ...

**Tone vs peers**
- <one or two sentences comparing tone>

**Implications for near-term price reaction**
- Bullet 1
- Bullet 2
- Bullet 3

Use ONLY the facts shown above. Do not assume external consensus, news, or valuations.
""".strip()


def historical_earnings_agent_prompt(
    fact: Dict[str, Any],
    related_facts: List[Dict[str, Any]],
    current_quarter: str | None = None,
) -> str:
    """Prompt template for comparing current commentary vs the firm's own past calls."""
    quarter_label = f"from {current_quarter}" if current_quarter else "for the current quarter"

    return f"""
You are an equity analyst specialising in comparing a company's **current earnings call**
with its **own past calls**.

Your goal is to understand whether current commentary represents an
**improvement, deterioration, or no major change** vs history, in a way that could
move the stock on the **next trading day**.

### Current fact {quarter_label}
{json.dumps(fact, indent=2, ensure_ascii=False)}

### Related historical facts (previous quarters)
{json.dumps(related_facts, indent=2, ensure_ascii=False)}

### Tasks
1. For this topic, identify the **most relevant 1-3 historical facts**.
2. For the underlying metric or theme, classify the trend as:
   - **Accelerating**, **Decelerating**, or **Broadly in line**.
3. Call out:
   - Where management is **more optimistic or more cautious** than before.
   - Where actual results **beat, meet, or miss** what they previously guided.

### Output structure

**Fact-level comparison**
- <Metric / topic>: <Accelerating / Decelerating / In line>
  - Evidence: <one short sentence referencing current vs past numbers/wording>

**Key historical context for near-term reaction**
- Bullet 1
- Bullet 2
- Bullet 3

Keep it concise and focus only on changes that could affect the **one-day** price reaction.
Use ONLY the information in the facts above.
""".strip()


def financials_statement_agent_prompt(
    fact: Dict[str, Any],
    similar_facts: list,
) -> str:
    """Prompt template for analysing quantitative changes in financials."""
    return f"""
You are an equity analyst specialising in **financial statements**.

Your ONLY objective is to interpret the **quantitative changes** in this company's
financials in a way that matters for the **near-term (one-trading-day) price reaction**.

### Inputs
- Current-quarter quantitative facts (JSON list or dict):
{json.dumps(fact, indent=2, ensure_ascii=False)}

- Most similar quantitative facts from previous quarters:
{json.dumps(similar_facts, indent=2, ensure_ascii=False)}

### Tasks
1. For each major metric (revenue, EPS, margins, cash flow, leverage, etc.):
   - Compare the current value to historical values.
   - Classify the change as **Better**, **Worse**, or **Broadly in line**.
2. Highlight **large changes** (e.g. >10% growth swing, >5 percentage-point margin delta)
   and state whether they are likely **positive or negative** for the **near-term** price.
3. Point out any **one-off items** or unsustainable drivers if they are explicitly mentioned.

### Output structure

**Metric-level assessment**
- <Metric>: <Better / Worse / In line>, <one short sentence with numbers or direction>

**Key financial takeaways for near-term price reaction**
- Bullet 1
- Bullet 2
- Bullet 3

Be concise. Use ONLY the numbers and facts shown in the inputs.
Do NOT guess or fabricate figures that are not provided.
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
You previously wrote the following research note for this company:

<Previous note>
{all_notes}

After that call, the stock moved {actual_return} on the first trading day.

When forming your new view, briefly reflect on whether your previous note was
too optimistic or too pessimistic relative to the actual move, and slightly
adjust your Direction score if appropriate. Do NOT over-correct; use this
only as a mild calibration signal.
""".strip()


def main_agent_prompt(
    notes,
    all_notes = None,
    original_transcript: str | None = None,
    memory_txt: str | None = None,
    financial_statements_facts: str | None = None,
    qoq_section: str | None = None,
) -> str:
    """
    Combine helper notes and optional transcript into the final PM verdict prompt.

    Parameters
    ----------
    notes : dict
        Typically shaped like {"financials": "...", "past": "...", "peers": "..."}.
    all_notes : Any
        Reserved for pipeline compatibility.
    original_transcript : str | None
        Full or truncated earnings call transcript.
    memory_txt : str | None
        Calibration memory generated by `memory`.
    financial_statements_facts : str | None
        Additional quantitative facts (YoY).
    qoq_section : str | None
        Quarter-on-quarter facts if available.
    """
    transcript_section = ""
    if original_transcript:
        transcript_section = f"""
---
Latest earnings call transcript (full, not pre-summarised or truncated):
{original_transcript}
---"""

    financial_statements_section = ""
    if financial_statements_facts:
        financial_statements_section = f"""
---
Financial Statements Facts (YoY):
{financial_statements_facts}
---"""

    qoq_section_str = ""
    if qoq_section:
        qoq_section_str = f"""
---
Quarter-on-Quarter Changes:
{qoq_section}
---"""

    calibration_section = ""
    if memory_txt:
        calibration_section = f"""
---
Calibration memory:
{memory_txt}
---"""

    return f"""
You are a long-only portfolio manager focused on the **one-trading-day price move
after an earnings call**.

Your ONLY objective is to judge the likely **one-trading-day price reaction** after
the earnings call, not long-term fundamentals or multi-month performance.

Your job:
1. Read the three specialist notes below (financials, history, peers).
2. Combine them into a single view of how the market is likely to react **tomorrow**.
3. Output a concise explanation plus a **Direction score from 0 to 10**.

### Specialist notes
- **Financials vs history (YoY/QoQ, margins, cash flow)**  
{notes.get('financials', '')}

- **Company vs its own past calls (guidance credibility & narrative shifts)**  
{notes.get('past', '')}

- **Company vs peers (relative growth, margins, and tone)**  
{notes.get('peers', '')}
{financial_statements_section}
{qoq_section_str}
{calibration_section}
{transcript_section}

### Decision rules
- Optimise for the **next trading day reaction**, not long-term fundamentals.
- Give the most weight to:
  1. Positive/negative **surprises vs recent trend and expectations**,
  2. Changes in **guidance, demand, and profitability**, and
  3. Whether the company looks **better or worse than peers**.
- If signals conflict, prioritise guidance and clear surprises over small numerical noise.
- Use ONLY the information above; do NOT invent external data (e.g. consensus or news).

### Direction scale (must be consistent with your reasoning)
- 0-3  = you expect a clear **Down** move.
- 4-6  = reaction is likely **flat / very uncertain**.
- 7-10 = you expect a clear **Up** move.

### Output format (MUST FOLLOW EXACTLY)

1. First, write **2-3 sentences** explaining:
   - Why you expect an Up / Down / almost flat reaction,
   - Refer explicitly to financials, past calls, and peer positioning where relevant.

2. Then, on a new line, output the Direction score as an integer:

Direction: <integer 0-10>

Example (structure only):

Management cut guidance and highlighted demand headwinds, while peers are
more optimistic. This makes a negative one-day reaction more likely.
Direction: 3
""".strip()


def facts_extraction_prompt(transcript_chunk: str) -> str:
    """
    Prompt for chunk-based fact extraction.
    """
    return f"""
You are a senior equity-research analyst extracting structured facts from an
earnings call transcript.

Your ONLY objective is to identify as many **distinct, decision-relevant facts**
as possible from the text below. Facts should be things like:
- Reported results (revenue, EPS, margins, cash flow, segment performance)
- Forward-looking statements (guidance, outlook, demand commentary, capex plans)
- Risks and uncertainties (headwinds, competition, regulatory issues)
- Sentiment / tone (confidence, caution, repeated emphasis)
- Macro or industry-wide observations that affect the company

The transcript chunk is:
---
{transcript_chunk}
---

### How to extract facts

- Split the text into **30-70 concise facts** if possible; do not merge unrelated points.
- Each fact should focus on **one main metric or theme**.
- Use the most specific metric name you can (e.g. "Cloud revenue growth" not just "growth").

### OUTPUT FORMAT (STRICT)

Return ONLY a sequence of markdown blocks in this exact form, with no extra text
before, between, or after the blocks:

### Fact No. 1
- **Type:** <Result | Forward-Looking | Risk Disclosure | Sentiment | Macro>
- **Metric:** <short metric name>
- **Value:** <numeric or qualitative value exactly as stated, if possible>
- **Context:** <short explanation of why this fact matters>

### Fact No. 2
- **Type:** ...
- **Metric:** ...
- **Value:** ...
- **Context:** ...

...

Rules:
- You MUST output more than 30 facts if the text is long enough.
- Do NOT include any anonymisation tags like [ORG]; use plain company / product names.
- Do NOT add commentary outside the specified fields.
- Use ONLY information from the transcript chunk above; do not invent numbers or facts.
""".strip()


def facts_delegation_prompt(facts: List) -> str:
    """
    Prompt for assigning facts to helper tools.
    """
    facts_json = json.dumps(facts, indent=2, ensure_ascii=False)

    return f"""
You are the RAG-orchestration analyst for an earnings-call workflow.

You are given a list of extracted facts from the latest earnings call. For each
fact, you must decide which retrieval tools would add useful context for
predicting the **one-trading-day price reaction** after the call.

Here are the extracted facts (JSON list or dict):
{facts_json}

Available tools:
1. InspectPastStatements  - pull prior financial statement data around this metric.
2. QueryPastCalls         - pull this company's own past commentary on this topic.
3. CompareWithPeers       - pull similar metrics from comparable firms.

Some facts may benefit from multiple tools; others may not need any extra context.

### OUTPUT FORMAT (STRICT)

Output exactly three lines, and nothing else:

InspectPastStatements: Fact No <comma-separated integers, or empty if none>
QueryPastCalls: Fact No <comma-separated integers, or empty if none>
CompareWithPeers: Fact No <comma-separated integers, or empty if none>

Examples:
InspectPastStatements: Fact No 2, 4, 6
QueryPastCalls: Fact No 1, 3, 5
CompareWithPeers: Fact No 10

or, if a tool is not needed:

InspectPastStatements: Fact No
QueryPastCalls: Fact No 1, 2
CompareWithPeers: Fact No 3

Rules:
- Use ONLY the facts shown above.
- Prefer InspectPastStatements for **numerical metrics** (revenue, margins, cash flow).
- Prefer QueryPastCalls for **guidance, strategy, and management narrative**.
- Prefer CompareWithPeers for **relative performance or competitive positioning**.
- Do NOT explain your choices. Do NOT add any other text.
""".strip()


peer_discovery_ticker_prompt = """
You are a financial analyst.

Given a public company with ticker {ticker}, identify 5 close peer companies
whose primary businesses are in the same or very closely related industries.

Selection rules:
Focus on companies whose core business is in the same value chain
 (similar products, services, and end-markets), not just the same GICS sector.
Prefer companies listed in the U.S. or as ADRs when possible.
Prefer similar business model and revenue mix (e.g., electronic test and
measurement equipment, instrumentation, or closely related solutions).
Avoid ETFs, indices, funds, preferred shares, warrants, and companies that
have been acquired/delisted.
Market cap and scale should be broadly comparable (roughly within
 0.25x-4x of the target company, if possible).

Only output a Python-style list of 5 unique ticker symbols, with no explanation, like:
["AAPL", "GOOGL", "AMZN", "MSFT", "ORCL"]
"""


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
