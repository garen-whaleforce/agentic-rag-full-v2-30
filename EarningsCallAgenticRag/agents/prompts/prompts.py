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
    "get_default_system_prompts",
    "get_all_default_prompts",
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

# ============================================================================
# SYSTEM MESSAGES (6)
# ============================================================================

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
# PROMPT TEMPLATES (9)
# Use {{variable}} placeholders - will be replaced at runtime
# ============================================================================

_DEFAULT_COMPARATIVE_AGENT_PROMPT = """
You are analyzing a company's earnings call transcript alongside statements made by similar firms.{{ticker_section}}

The batch of facts about the firm is:
{{facts}}

Comparable firms discuss the facts in the following way:
{{related_facts}}

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


_DEFAULT_HISTORICAL_EARNINGS_AGENT_PROMPT = """
You are analyzing a company's earnings call transcript alongside facts from its own past earnings calls.

The list of current facts are:
{{fact}}

It is reported in the quarter {{quarter_label}}

Here is a JSON list of related facts from the firm's previous earnings calls:
{{related_facts}}

TASK
────
1. **Validate past guidanced**
   ▸ For every forward-looking statement made in previous quarters, state whether the firm met, beat, or missed that guidance in `{{quarter_label}}`.
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


_DEFAULT_FINANCIALS_STATEMENT_AGENT_PROMPT = """
You are reviewing the company's {{quarter_label}} earnings-call transcript and comparing a key fact to the most similar historical facts from previous quarters.

────────────────────────────────────────
Current fact (from {{quarter_label}}):
{{fact}}

Most similar past facts (from previous quarters):
{{similar_facts}}
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


_DEFAULT_MAIN_AGENT_PROMPT = """
You are a portfolio manager focusing on very short-term post-earnings reactions (the next trading day).

You are given:
- The original earnings call transcript.{{transcript_section}}
- Structured notes comparing this quarter's results and guidance versus the company's own history and versus peers.
- Key financial statement excerpts and year-on-year changes.

Your job is to decide whether the stock price is likely to **increase ("Up") or decrease ("Down") one trading day after the earnings call**, and to assign a **Direction score from 0 to 10**.

Use the information below:

Original transcript:
{{original_transcript}}

{{financial_statements_section}}
{{qoq_section_str}}
---
Financials-vs-History note:
{{notes_financials}}

Historical-Calls note:
{{notes_past}}

Peers note:
{{notes_peers}}

{{memory_section}}

---

### Step 1 – Classify management tone

From the transcript and notes, classify management's **near-term tone (next 1–3 quarters)** as exactly one of:
- Very optimistic
- Moderately optimistic
- Balanced
- Moderately cautious
- Very cautious

Base this on concrete wording (for example: "strong demand", "robust pipeline", "headwinds", "macro uncertainty", "taking a conservative stance", "softening", "slower than expected", "we are cautious").

Explicitly mention this tone classification in your explanation.

---

### Step 2 – Compare results and guidance versus investor expectations

Infer how the **current quarter and guidance** compare to what investors were likely expecting, using these heuristics:

- Positive surprise (bullish for next-day reaction):
  - Clear beat of prior guidance or prior trends **and** management raises or tightens guidance upward.
  - Growth is accelerating versus recent quarters or versus peers.
  - Mix shift or margin trends clearly improve the quality of earnings.

- Negative surprise (bearish for next-day reaction):
  - Guidance is cut or framed more conservatively; growth is slowing or margins are under pressure.
  - Management emphasizes macro headwinds, demand softness, elongated deal cycles, or higher uncertainty.
  - Results are only "in line" with previous guidance after a strong run-up in prior quarters.

- Mixed:
  - Strong reported numbers but guidance is cautious or only in-line.
  - Solid top-line but with weakening profitability or cash flow.
  - Beat driven by one-offs that are unlikely to repeat.

When positives and negatives conflict, for **next-day price reaction** the change in **forward guidance and tone** usually dominates realised results.
Do **not** assume "beat and raise" always means the stock goes up sharply if guidance is only slightly better or the tone is cautious.

#### Special case 1 – "Record quarter but slowing going forward"

If the company reports record results or a strong beat, but:
- revenue, bookings, or key growth metrics are clearly **decelerating** versus recent quarters, or
- full-year / next-quarter guidance implies **slower growth or lower margins** going forward,

then you must significantly **discount** the bullish impact of the "record" numbers.

In such cases:
- Do **not** assign scores of 8–10.
- If growth deceleration or softer forward guidance is the main new information, you should lean **Neutral to slightly Negative** (Direction **4–6 at most**, and often **4–5**), even if the current quarter looks very strong in isolation.

#### Special case 2 – "Relief rally / less-bad than feared"

If recent quarters have been weak or under pressure and:
- management now shows clear signs of **stabilization or bottoming**, or
- guidance is merely "in line" but clearly **less bad than investors previously feared**, or
- major risks (liquidity, leverage, execution, product issues, regulatory overhangs) are **de-risked**,

treat this as a potential **"relief rally"** setup.

In these situations, avoid assigning very negative scores (0–2) unless new information is clearly **worse** than what investors were already worried about. Mixed but improving situations should lean toward **5–7** rather than **2–4**.

---

### Step 3 – Assign Direction score (0–10)

Use the full scale consistently for the **next trading day** reaction:

- **0–2**: High conviction of a meaningful **down** reaction (often ≥10% drop).
  - Requires clearly negative surprise and/or very cautious tone.
- **3–4**: Mildly negative; risk skewed to the downside, but not a disaster.
- **5**: Balanced or very unclear; upside and downside are similar. Avoid forcing a call.
- **6–7**: Mildly positive; some upside but not a big re-rating.
- **8–10**: High conviction of a strong **up** reaction (often ≥10% rise), driven by **both** a clear positive surprise **and** confident tone.

Be **conservative**:
- Do **not** give scores ≥8 just because this quarter is "strong" in absolute terms.
- Reserve scores ≥8 for cases where guidance and tone clearly **raise** investor expectations.
- If there are meaningful headwinds, signs of deceleration, or conservative guidance, cap the score at **6** even if the reported quarter looks strong.

#### Calibration of Direction scores

- Scores **0–1** and **9–10** should be rare and reserved only for extreme, very clear cases.
- Scores **2–3** and **7–8** represent high-conviction negative or positive reactions.
- Scores **4** and **6** are for weak, low-conviction tilts only and should be used sparingly.

Use this rule:

- If you can list at least **three independent bullish drivers** and at most one minor bearish driver,
  you should use a Direction score of **7 or higher** (not 6).
- If you can list at least **three independent bearish drivers** and at most one minor bullish driver,
  you should use a Direction score of **3 or lower** (not 4).

- If you can list two or more meaningful drivers on **each side** (both bullish and bearish),
  the situation is truly mixed: use a Direction score of **5 (Neutral)**.

- Use a Direction of **4** only when the bear case clearly dominates but you still see some material positives.
- Use a Direction of **6** only when the bull case clearly dominates but you still see some material risks.

If you are unsure whether to pick 4/6 or 3/7, be honest and pick **5 (Neutral)** instead.

---

### Step 4 – Final verdict (short-term, next trading day)

1. Briefly list:
   - The main drivers that could push the stock **up** tomorrow.
   - The main drivers that could push the stock **down** tomorrow.
   Then state which side you believe dominates the **next-day** move and why.
2. Make sure your Direction score is consistent with your reasoning
   (for example, if you highlight several serious risks and a cautious tone, avoid a very high score).

---

Respond in **exactly** this format in English:

<Couple of sentences of Explanation in English, including tone classification and key up/down drivers>

**Summary: <Two sentences supporting your verdict with facts and evidence>, Direction : <0-10>**

Bilingual requirement:
- First, produce the full output in **English** exactly following the format specified above, including the final "Direction" line.
- After you have produced the English output, append a second section where you
  translate the entire explanation (but not the numeric score) into Traditional Chinese (繁體中文), preserving the same reasoning and structure.
- Do NOT modify the English output or its format; the Chinese section must come AFTER the English output.
""".strip()


_DEFAULT_FACTS_EXTRACTION_PROMPT = """
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

The transcript is {{transcript_chunk}}

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

""".strip()


_DEFAULT_FACTS_DELEGATION_PROMPT = """
You are the RAG-orchestration analyst for an earnings-call workflow.

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
The facts are: {{facts}}

Output your answers in the following form:

InspectPastStatements: Fact No <2, 4, 6>
CompareWithPeers:  Fact No <10>
QueryPastCalls: Fact No <1, 3, 5>

*One fact may appear under multiple tools if multiple comparisons are helpful.*

""".strip()


_DEFAULT_PEER_DISCOVERY_TICKER_PROMPT = """
You are a financial analyst. Based on the company with ticker {{ticker}}, list 5 close peer companies that are in the same or closely related industries.

Only output a Python-style list of tickers, like:
["AAPL", "GOOGL", "AMZN", "MSFT", "ORCL"]
""".strip()


_DEFAULT_MEMORY_PROMPT = """
You have memory on how your previous prediction on the firm faired.
Your previous research note is given as:
{{all_notes}},
The actual return achieved by your previous note was : {{actual_return}}
""".strip()


_DEFAULT_BASELINE_PROMPT = """
You are a portfolio manager and you are reading an earnings call transcript.
decide whether the stock price is likely to **increase ("Up") or decrease ("Down")**
one trading day after the earnings call, and assign a **Direction score** from 0 to 10.

---
{{transcript}}

Instructions:
1. Assign a confidence score (0 = strong conviction of decline, 5 = neutral, 10 = strong conviction of rise).

Respond in **exactly** this format:

<Couple of sentences of Explanation>
Direction : <0-10>
""".strip()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_default_system_prompts() -> Dict[str, str]:
    """
    回傳六個 system prompt 的「原始 default 版本」。
    key 為 EDITABLE_PROMPT_KEYS 中使用的 key。
    """
    return {
        "MAIN_AGENT_SYSTEM_MESSAGE": _DEFAULT_MAIN_AGENT_SYSTEM_MESSAGE,
        "EXTRACTION_SYSTEM_MESSAGE": _DEFAULT_EXTRACTION_SYSTEM_MESSAGE,
        "DELEGATION_SYSTEM_MESSAGE": _DEFAULT_DELEGATION_SYSTEM_MESSAGE,
        "COMPARATIVE_SYSTEM_MESSAGE": _DEFAULT_COMPARATIVE_SYSTEM_MESSAGE,
        "HISTORICAL_EARNINGS_SYSTEM_MESSAGE": _DEFAULT_HISTORICAL_EARNINGS_SYSTEM_MESSAGE,
        "FINANCIALS_SYSTEM_MESSAGE": _DEFAULT_FINANCIALS_SYSTEM_MESSAGE,
    }


def get_all_default_prompts() -> Dict[str, str]:
    """
    回傳全部 15 個 prompt 的「原始 default 版本」。
    包含 6 個 system messages + 9 個 prompt templates。
    """
    return {
        # System Messages (6)
        "MAIN_AGENT_SYSTEM_MESSAGE": _DEFAULT_MAIN_AGENT_SYSTEM_MESSAGE,
        "EXTRACTION_SYSTEM_MESSAGE": _DEFAULT_EXTRACTION_SYSTEM_MESSAGE,
        "DELEGATION_SYSTEM_MESSAGE": _DEFAULT_DELEGATION_SYSTEM_MESSAGE,
        "COMPARATIVE_SYSTEM_MESSAGE": _DEFAULT_COMPARATIVE_SYSTEM_MESSAGE,
        "HISTORICAL_EARNINGS_SYSTEM_MESSAGE": _DEFAULT_HISTORICAL_EARNINGS_SYSTEM_MESSAGE,
        "FINANCIALS_SYSTEM_MESSAGE": _DEFAULT_FINANCIALS_SYSTEM_MESSAGE,
        # Prompt Templates (9)
        "COMPARATIVE_AGENT_PROMPT": _DEFAULT_COMPARATIVE_AGENT_PROMPT,
        "HISTORICAL_EARNINGS_AGENT_PROMPT": _DEFAULT_HISTORICAL_EARNINGS_AGENT_PROMPT,
        "FINANCIALS_STATEMENT_AGENT_PROMPT": _DEFAULT_FINANCIALS_STATEMENT_AGENT_PROMPT,
        "MAIN_AGENT_PROMPT": _DEFAULT_MAIN_AGENT_PROMPT,
        "FACTS_EXTRACTION_PROMPT": _DEFAULT_FACTS_EXTRACTION_PROMPT,
        "FACTS_DELEGATION_PROMPT": _DEFAULT_FACTS_DELEGATION_PROMPT,
        "PEER_DISCOVERY_TICKER_PROMPT": _DEFAULT_PEER_DISCOVERY_TICKER_PROMPT,
        "MEMORY_PROMPT": _DEFAULT_MEMORY_PROMPT,
        "BASELINE_PROMPT": _DEFAULT_BASELINE_PROMPT,
    }


# ============================================================================
# PROMPT FUNCTIONS
# ============================================================================


def comparative_agent_prompt(
    facts: List[Dict[str, Any]],
    related_facts: List[Dict[str, Any]],
    self_ticker: str | None = None,
) -> str:
    """Return the prompt for the *Comparative Peers* analysis agent."""
    template = get_prompt_override("COMPARATIVE_AGENT_PROMPT", _DEFAULT_COMPARATIVE_AGENT_PROMPT)
    ticker_section = f"\n\nThe ticker of the firm being analyzed is: {self_ticker}" if self_ticker else ""

    return template.replace(
        "{{ticker_section}}", ticker_section
    ).replace(
        "{{facts}}", json.dumps(facts, indent=2)
    ).replace(
        "{{related_facts}}", json.dumps(related_facts, indent=2)
    )


def historical_earnings_agent_prompt(
    fact: Dict[str, Any],
    related_facts: List[Dict[str, Any]],
    current_quarter: str | None = None,
) -> str:
    """Return the prompt for the *Historical Earnings* analysis agent."""
    template = get_prompt_override("HISTORICAL_EARNINGS_AGENT_PROMPT", _DEFAULT_HISTORICAL_EARNINGS_AGENT_PROMPT)
    quarter_label = current_quarter if current_quarter else "the current quarter"

    return template.replace(
        "{{fact}}", json.dumps(fact, indent=2)
    ).replace(
        "{{related_facts}}", json.dumps(related_facts, indent=2)
    ).replace(
        "{{quarter_label}}", quarter_label
    )


def financials_statement_agent_prompt(
    fact: Dict[str, Any],
    similar_facts: list,
    quarter: str | None = None,
) -> str:
    """Prompt template for analysing the current fact in the context of most similar past facts."""
    template = get_prompt_override("FINANCIALS_STATEMENT_AGENT_PROMPT", _DEFAULT_FINANCIALS_STATEMENT_AGENT_PROMPT)
    quarter_label = quarter if quarter else "the current quarter"

    return template.replace(
        "{{quarter_label}}", quarter_label
    ).replace(
        "{{fact}}", json.dumps(fact, indent=2)
    ).replace(
        "{{similar_facts}}", json.dumps(similar_facts, indent=2)
    )


def memory(all_notes, actual_return):
    """Combine prior note and realized move for calibration."""
    template = get_prompt_override("MEMORY_PROMPT", _DEFAULT_MEMORY_PROMPT)

    return template.replace(
        "{{all_notes}}", str(all_notes)
    ).replace(
        "{{actual_return}}", str(actual_return)
    )


def main_agent_prompt(
    notes,
    all_notes=None,
    original_transcript: str | None = None,
    memory_txt: str | None = None,
    financial_statements_facts: str | None = None,
    qoq_section: str | None = None,
) -> str:
    """Prompt for the *Main* decision-making agent."""
    template = get_prompt_override("MAIN_AGENT_PROMPT", _DEFAULT_MAIN_AGENT_PROMPT)

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

    return template.replace(
        "{{transcript_section}}", transcript_section
    ).replace(
        "{{original_transcript}}", original_transcript or ""
    ).replace(
        "{{financial_statements_section}}", financial_statements_section
    ).replace(
        "{{qoq_section_str}}", qoq_section_str
    ).replace(
        "{{notes_financials}}", notes.get('financials', '') if notes else ''
    ).replace(
        "{{notes_past}}", notes.get('past', '') if notes else ''
    ).replace(
        "{{notes_peers}}", notes.get('peers', '') if notes else ''
    ).replace(
        "{{memory_section}}", memory_section
    )


def facts_extraction_prompt(transcript_chunk: str) -> str:
    """Build the LLM prompt that asks for five specific data classes."""
    template = get_prompt_override("FACTS_EXTRACTION_PROMPT", _DEFAULT_FACTS_EXTRACTION_PROMPT)

    return template.replace("{{transcript_chunk}}", transcript_chunk)


def facts_delegation_prompt(facts: List) -> str:
    """Return the prompt used for routing facts to helper tools."""
    template = get_prompt_override("FACTS_DELEGATION_PROMPT", _DEFAULT_FACTS_DELEGATION_PROMPT)

    return template.replace("{{facts}}", str(facts))


def peer_discovery_ticker_prompt(ticker: str) -> str:
    """Return the prompt for peer discovery."""
    template = get_prompt_override("PEER_DISCOVERY_TICKER_PROMPT", _DEFAULT_PEER_DISCOVERY_TICKER_PROMPT)

    return template.replace("{{ticker}}", ticker)


def baseline_prompt(transcript) -> str:
    """Return the baseline prompt for simple analysis."""
    template = get_prompt_override("BASELINE_PROMPT", _DEFAULT_BASELINE_PROMPT)

    return template.replace("{{transcript}}", str(transcript))
