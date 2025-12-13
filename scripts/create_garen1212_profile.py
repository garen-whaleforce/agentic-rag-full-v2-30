#!/usr/bin/env python3
"""
Create garen1212 profile with T+30 optimized MAIN_AGENT_PROMPT and FACTS_EXTRACTION_PROMPT.

Key changes from original:
- MAIN_AGENT_PROMPT: Changed from D+1 to T+30 (30 trading days) prediction horizon
- FACTS_EXTRACTION_PROMPT: Added T+30 selection criteria and minimum quotas per category

All other prompts remain unchanged from the base profile (garen1207).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from storage import get_prompt_profile, set_prompt_profile


# ==============================================================================
# NEW_MAIN_AGENT_PROMPT - Optimized for T+30 (30 Trading Days) Prediction
# ==============================================================================
NEW_MAIN_AGENT_PROMPT = """You are a portfolio manager analyzing an earnings call transcript.
Your objective is to predict the stock price direction **over the next 30 trading days (T+30)**
following the earnings call.{{transcript_section}}

Assign a **Direction score from 0 to 10** based on your conviction.

The original transcript is:

{{original_transcript}}

{{financial_statements_section}}
{{qoq_section_str}}
---
Financials-vs-History note:
{{notes_financials}}

Historical-Calls note:
{{notes_past}}

Peer-Comparison note:
{{notes_peers}}

{{memory_section}}

---

## TIME HORIZON: 30 TRADING DAYS (T+30)

**Critical**: Your prediction is for **30 trading days** after earnings, NOT the next day.
Over this longer horizon, the following factors matter most:

1. **Forward Guidance Quality**
   - Is guidance raised, maintained, or cut vs. prior quarter?
   - How specific vs. vague is the guidance?
   - Does guidance imply acceleration or deceleration in growth?

2. **Demand & Revenue Trends**
   - Is demand strengthening, stable, or weakening?
   - Are there signs of order slowdown, elongated sales cycles, or customer hesitation?
   - Regional/segment mix shifts that may persist

3. **Margin & Cost Structure**
   - Gross margin trajectory (pricing power vs. input cost pressure)
   - Operating leverage: Are costs being controlled or expanding faster than revenue?
   - One-time vs. structural margin changes

4. **Inventory & Channel Health**
   - Inventory build-up (risk of future write-downs or discounting)
   - Channel stuffing concerns or healthy sell-through
   - Days inventory outstanding trends

5. **Capital Allocation & Liquidity**
   - Free cash flow generation and conversion
   - Debt levels and interest coverage
   - Buyback announcements or dividend changes
   - CapEx intensity vs. peers

6. **Risk Disclosures**
   - Regulatory, litigation, or compliance issues
   - Competitive threats explicitly mentioned
   - Supply chain or geopolitical risks
   - Currency headwinds quantified

7. **Notes Integration**
   - How do financials-vs-history, historical calls, and peer comparison notes
     inform the 30-day outlook?

---

## DIRECTION SCORING RUBRIC (0-10)

**MANDATORY**: Start from 5 (neutral) and adjust based on evidence. Do NOT default to optimism.

### Scoring Guidelines:

**0-3 (DOWN - Bearish over T+30):**
- Score **0-1**: Catastrophic outlook. Multiple severe negatives: guidance cut significantly,
  demand collapse, major margin compression, liquidity concerns, or serious regulatory/legal risk.
- Score **2-3**: Clearly negative. At least ONE major negative driver:
  - Guidance cut or significantly below expectations
  - Demand weakening or growth decelerating meaningfully
  - Margin pressure without clear resolution path
  - Cash flow deterioration or rising leverage concerns
  - Competitive position worsening
  - Inventory/channel problems signaled

**4-6 (NEUTRAL):**
- Score **4**: Slight negative lean. Mixed signals but risks outweigh positives.
- Score **5**: Truly balanced. Cannot determine directional edge with confidence.
  Use when evidence is contradictory or consists mostly of generic statements without
  specific numbers or commitments.
- Score **6**: Slight positive lean. Mixed signals but positives marginally outweigh risks.

**7-10 (UP - Bullish over T+30):**
- Score **7-8**: Clearly positive. Requires AT LEAST TWO independent, evidence-backed
  positive drivers for the next 30 days:
  - Guidance raised with specific numbers
  - Demand acceleration with concrete proof (order growth, backlog, bookings)
  - Margin expansion with sustainable drivers
  - Strong FCF generation and shareholder-friendly capital allocation
  - Competitive wins documented
- Score **9-10**: Exceptional outlook. Three or more strong positive drivers,
  minimal risks, and high management credibility. Very rare.

### Anti-Bullish-Bias Rules:

1. **Do NOT give 7+ if:**
   - Guidance is merely "maintained" or "in-line"
   - Results beat but guidance is unchanged or vague
   - Management uses excessive promotional language without numbers
   - Growth is decelerating even if absolute numbers are good
   - Margin or FCF trends are flat or negative

2. **MUST give 0-3 if any ONE of these is present:**
   - Guidance is cut (revenue or EPS)
   - Explicit mention of demand softness, order slowdown, or pipeline weakness
   - Gross margin declined YoY and QoQ with no clear recovery plan
   - Cash flow negative or significantly below prior periods
   - New material risk disclosed (litigation, regulatory, competitive threat)
   - Inventory days increased significantly

3. **Default to 4-6 (NEUTRAL) if:**
   - Evidence is mixed with meaningful positives AND negatives
   - Statements are qualitative without specific numbers
   - "Record quarter" but forward outlook is cautious or unchanged
   - Beat on results but missed on guidance or vice versa

---

## REQUIRED OUTPUT FORMAT

Produce your analysis in **exactly** this format:

<Explanation in 2-4 sentences covering:
1. The primary T+30 driver (positive or negative) with specific evidence
2. The main risk that could work against your thesis
Keep explanation factual with numbers or direct quotes where possible.>

**Summary: <Two sentences supporting your verdict with facts and evidence>, Direction : <0-10>**

---

## BILINGUAL REQUIREMENT

- First, produce the full output in **English** exactly following the format above,
  including the final "**Summary: ..., Direction : <0-10>**" line.
- After the English output, append a section where you translate the entire explanation
  (but NOT the Summary line or Direction score) into Traditional Chinese (繁體中文),
  preserving the same reasoning and structure.
- Do NOT modify the English output or its format; the Chinese section must come AFTER
  the English output.
- Do NOT write a second "Direction" line in any language.
""".strip()


# ==============================================================================
# NEW_FACTS_EXTRACTION_PROMPT - Optimized for T+30 Relevant Facts
# ==============================================================================
NEW_FACTS_EXTRACTION_PROMPT = """You are a senior equity-research analyst extracting facts
that are most likely to impact stock price over the **next 30 trading days (T+30)**.

### TASK

Extract **only** the following five classes from the transcript below.
Ignore moderator chatter, safe-harbor boiler-plate, and anything that doesn't match one of these classes.

1. **Result** – Already-achieved financial or operating results
   Focus on: Revenue, EPS, margins (gross/operating/net), segment performance,
   cash flow, bookings, backlog, inventory levels, regional breakdown.

2. **Forward-Looking** – Any explicit future projection, target, plan, or guidance
   Focus on: Next quarter/full year guidance, demand commentary, pipeline,
   pricing expectations, margin outlook, CapEx plans, hiring/headcount,
   product launches, geographic expansion.
   **Priority**: Statements with specific timeframes (next quarter, H2, full year, near-term)

3. **Risk Disclosure** – Statements highlighting current or expected obstacles
   Focus on: Demand softness, order slowdown, pricing pressure, margin headwinds,
   inventory concerns, supply chain issues, FX headwinds, regulatory/legal risks,
   competitive threats, customer concentration, macro sensitivity.
   **CRITICAL**: Extract ALL risk-related statements. Do not skip negatives.

4. **Sentiment** – Management's overall tone
   Focus on: Confidence level, caution indicators, use of hedging language,
   comparison to prior quarter tone, specific word choices that signal
   optimism ("strong", "robust", "accelerating") or caution ("challenging",
   "uncertain", "softening", "slower than expected").

5. **Macro** – Discussion of macro-economic landscape impact
   Focus on: Interest rate sensitivity, inflation impact, consumer spending trends,
   enterprise spending environment, sector-specific cycles, trade/tariff effects.

The transcript is {{transcript_chunk}}

---

### MINIMUM QUOTAS (MANDATORY)

You MUST extract facts meeting these minimum counts:
- **Forward-Looking**: >= 12 facts
- **Risk Disclosure**: >= 10 facts
- **Result**: >= 8 facts
- **Sentiment**: >= 2 facts
- **Macro**: >= 2 facts

**Total**: You MUST output more than 30 facts (ideally 30-70).

If the transcript does not contain enough content for a category, still extract
the required number by:
- Breaking down compound statements into separate facts
- Noting "not explicitly mentioned" in Reason if inferring from context
- Extracting related implied risks even if not directly stated

**CRITICAL**: Do NOT under-extract Risk Disclosure. If management discusses ANY
challenge, headwind, concern, uncertainty, or negative trend, it MUST be captured.

---

### T+30 SELECTION CRITERIA

Prioritize facts that could materially impact stock price over 30 trading days:

**High Priority (extract these first):**
- Guidance changes (raised/lowered/maintained) with numbers
- Demand signals (acceleration/deceleration, pipeline, orders, bookings)
- Margin trajectory (expansion/compression, drivers)
- Inventory and channel health
- Cash flow and capital allocation (buybacks, dividends, debt)
- Competitive positioning changes
- Regulatory or legal developments
- Pricing power indicators

**Medium Priority:**
- Segment/regional performance breakdowns
- Product mix shifts
- Operational efficiency initiatives
- Cost structure changes
- Customer wins/losses

**Lower Priority (extract only if quota not met):**
- General market commentary without company-specific impact
- Long-term vision statements (>1 year horizon)
- Historical context without forward implications

---

### OUTPUT RULES

* Use the exact markdown block below for **every** extracted item.
* Increment the item number sequentially (1, 2, 3 …).
* One metric per block; never combine multiple metrics.
* **Metric** should be specific and include timeframe when available
  (e.g., "Q4 Revenue Guidance", "FY25 Gross Margin Outlook", "Near-term Demand")
* **Value** should capture exact numbers, ranges, or key phrases from transcript
* **Reason** should briefly explain why this fact matters for T+30 outlook

Do not include [ORG] in your response.

---

### EXAMPLE OUTPUT

### Fact No. 1
- **Type:** Result
- **Metric:** Q3 Revenue
- **Value:** "$4.2 billion, up 15% YoY"
- **Reason:** Strong topline growth indicates demand momentum that may persist into Q4.

### Fact No. 2
- **Type:** Forward-Looking
- **Metric:** Q4 Revenue Guidance
- **Value:** "$4.0-4.1 billion, implying 8-10% YoY growth"
- **Reason:** Guidance implies growth deceleration from Q3's 15%, which could pressure stock over T+30.

### Fact No. 3
- **Type:** Risk Disclosure
- **Metric:** Enterprise Demand Outlook
- **Value:** "Seeing elongated sales cycles and more cautious customer budgets"
- **Reason:** Suggests near-term revenue headwinds; negative signal for T+30 if trend continues.

### Fact No. 4
- **Type:** Risk Disclosure
- **Metric:** Gross Margin Pressure
- **Value:** "Expect 50-100bps gross margin headwind in Q4 from higher input costs"
- **Reason:** Quantified margin pressure is a negative catalyst for T+30 price action.

### Fact No. 5
- **Type:** Sentiment
- **Metric:** Management Tone - Cautious
- **Value:** "We are taking a prudent approach given macro uncertainty"
- **Reason:** Hedging language suggests management is de-risking expectations.

### Fact No. 6
- **Type:** Macro
- **Metric:** Consumer Spending Environment
- **Value:** "Seeing consumers trade down and reduce discretionary purchases"
- **Reason:** Weak consumer backdrop is a headwind for revenue if company has consumer exposure.
""".strip()


def main():
    from EarningsCallAgenticRag.agents.prompts.prompts import get_all_default_prompts

    # Try to read from garen1207 as base
    print("Reading garen1207 profile as base...")
    profile = get_prompt_profile("garen1207")

    if profile:
        print(f"Found garen1207 profile (updated_at: {profile.get('updated_at')})")
        prompts = profile.get("prompts", {})
        if not prompts:
            print("WARNING: garen1207 has no prompts defined, using defaults")
            prompts = get_all_default_prompts()
        print(f"Using {len(prompts)} prompts from garen1207 as base...")
    else:
        print("Profile 'garen1207' not found, using default prompts as base...")
        prompts = get_all_default_prompts()

    # Update ONLY the 2 prompts per user requirements
    print("\nUpdating prompts with T+30 optimized versions:")

    prompts["MAIN_AGENT_PROMPT"] = NEW_MAIN_AGENT_PROMPT
    print("  - MAIN_AGENT_PROMPT: UPDATED (T+30 horizon, anti-bullish-bias rubric)")

    prompts["FACTS_EXTRACTION_PROMPT"] = NEW_FACTS_EXTRACTION_PROMPT
    print("  - FACTS_EXTRACTION_PROMPT: UPDATED (T+30 selection criteria, minimum quotas)")

    # Save as garen1212
    print("\nSaving new profile 'garen1212'...")
    set_prompt_profile("garen1212", prompts)

    print("\n" + "=" * 70)
    print("SUCCESS: Created profile 'garen1212'")
    print("=" * 70)
    print(f"\nTotal prompts: {len(prompts)}")
    print("\nUpdated prompts (2 total):")
    print("  1. MAIN_AGENT_PROMPT")
    print("     - Changed prediction horizon from D+1 to T+30 (30 trading days)")
    print("     - Added detailed scoring rubric starting from 5 (neutral)")
    print("     - Implemented anti-bullish-bias rules")
    print("     - Requires 2+ independent evidence-backed drivers for scores 7+")
    print("     - Mandatory downgrade to 0-3 for any major negative signal")
    print("")
    print("  2. FACTS_EXTRACTION_PROMPT")
    print("     - Added T+30 selection criteria prioritization")
    print("     - Enforced minimum quotas:")
    print("       * Forward-Looking >= 12")
    print("       * Risk Disclosure >= 10")
    print("       * Result >= 8")
    print("       * Sentiment >= 2")
    print("       * Macro >= 2")
    print("     - Emphasized extracting ALL risk-related statements")
    print("     - Added timeframe-specific metric naming")
    print("")
    print("Unchanged prompts (inherited from garen1207):")
    print("  - MAIN_AGENT_SYSTEM_MESSAGE")
    print("  - COMPARATIVE_AGENT_PROMPT")
    print("  - HISTORICAL_EARNINGS_AGENT_PROMPT")
    print("  - FINANCIALS_STATEMENT_AGENT_PROMPT")
    print("  - All other system messages and prompts")
    print("")
    print("To apply this profile:")
    print("  POST /api/prompt_profiles/apply with body: {\"name\": \"garen1212\"}")
    print("  Or select 'garen1212' from the UI dropdown and click Apply")


if __name__ == "__main__":
    main()
