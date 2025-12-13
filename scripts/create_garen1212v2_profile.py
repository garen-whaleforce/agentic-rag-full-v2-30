#!/usr/bin/env python3
"""
Create garen1212v2 profile with BALANCED T+30 prompts.

Key fixes from garen1212:
- Changed from "trigger-based MUST DOWN" to "weighted net-score" approach
- Risk disclosures are now down-weighted (generic/boilerplate doesn't auto-trigger bearish)
- NEUTRAL (4-6) is the default for mixed/vague evidence
- Both UP and DOWN require concrete, citable evidence
- Scores 0-1 and 9-10 are rare extremes

All other prompts remain unchanged from the base profile (garen1207).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from storage import get_prompt_profile, set_prompt_profile


# ==============================================================================
# NEW_MAIN_AGENT_PROMPT - Balanced T+30 with net-score approach
# ==============================================================================
NEW_MAIN_AGENT_PROMPT = """You are a portfolio manager and you are reading an earnings call transcript.{{transcript_section}}
Decide whether the stock price is likely to **increase ("Up") or decrease ("Down")**
over the **NEXT 30 TRADING DAYS (T+30)** after the earnings call, and assign a **Direction score** from 0 to 10.

IMPORTANT: This is a 30-trading-day horizon task. Do NOT predict the next-day reaction.

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

Instructions:
1. Assign a Direction score (0 = strong conviction of decline over the next 30 trading days, 5 = neutral/mixed/unclear, 10 = strong conviction of rise over the next 30 trading days).
2. Evaluate all three notes together (Financials-vs-History, Historical-Calls, Peer-Comparison).
3. Consider the financial statements facts when available.
4. Pay special attention to year-on-year changes, especially bottom line figures (e.g., net profit), AND whether guidance/demand/margins imply near-term (30D) acceleration or deterioration.

MANDATORY SCORING RUBRIC (balanced; avoid systematic bearishness):
- Start from a BASE score of 5 (neutral). Do NOT default bullish or bearish.
- Identify up to 3 BULLISH 30D drivers and up to 3 BEARISH 30D drivers.
  Each driver MUST be supported by specific evidence from the transcript and/or the notes.
  Prefer numbers, explicit guidance changes, clear demand/margin statements, concrete actions (buybacks/capex), and near-term catalysts/risks.
- Assign each driver a strength:
  * Strong driver: +2 (bullish) or -2 (bearish) — requires explicit, concrete evidence likely to matter within ~30 trading days
    (e.g., raised/lowered guidance, clear demand inflection, margin expansion/compression with drivers, liquidity stress, major contract win/loss, regulatory event).
  * Mild driver: +1 or -1 — supported but less definitive / not clearly near-term.
- Compute: SCORE = 5 + (sum bullish strengths) + (sum bearish strengths), then clamp to [0..10] and round to an integer.

RISK DISCLOSURE WEIGHTING (to prevent over-correction):
- Many calls contain generic cautionary language. Do NOT treat generic or boilerplate risk mention as a strong bearish driver.
- Only count a risk as a bearish driver if it is (a) company-specific, (b) described as worsening/new or clearly material, AND (c) plausibly impacts the next ~30 trading days.
- If the evidence is mixed, vague, or mostly qualitative talking points, keep the score in the NEUTRAL range (4–6).

GUARDRAILS (rare extremes):
- Scores 0–1 or 9–10 must be rare and require very strong, unambiguous net evidence.
- If you cannot cite at least ONE concrete bullish driver, do NOT output 7–10.
- If you cannot cite at least ONE concrete bearish driver, do NOT output 0–3.
- If there are both meaningful bullish and bearish drivers, default to 4–6 unless one side clearly dominates.

Respond in **exactly** this format:

<Couple of sentences of Explanation>

**Summary: <Two sentences supporting your verdict with facts and evidence>, Direction : <0-10>**

Summary requirements (still exactly two sentences):
- Sentence 1: the single most important 30D driver (with specific evidence).
- Sentence 2: the single most important 30D risk/offset (with specific evidence).
- Keep the evidence grounded in transcript / notes. Avoid generic claims.

Bilingual requirement:
- First, produce the full output in **English** exactly following the format
  and examples specified above, including the final "Direction" line.
- After you have produced the English output, append a second section where you
  translate the entire explanation (but not the numeric score) into
  Traditional Chinese (繁體中文), preserving the same reasoning and structure.
- Do NOT modify the English output or its format; the Chinese section must come
  AFTER the English output.
- Do NOT write a second "Direction" line in any language.""".strip()


# ==============================================================================
# NEW_FACTS_EXTRACTION_PROMPT - Balanced with T+30 relevance filter
# ==============================================================================
NEW_FACTS_EXTRACTION_PROMPT = """You are a senior equity-research analyst.

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

IMPORTANT SELECTION PRINCIPLE (T+30 relevance):
Prioritize items that are likely to impact the NEXT ~30 TRADING DAYS:
- guidance changes, demand/bookings/backlog trends, margin drivers, pricing power, cost pressures,
  inventory/channel health, capex changes, buybacks/liquidity, major customer/contract events,
  competition intensity, regulatory/legal actions, and clearly stated near-term catalysts/risks.

RISK DISCLOSURE FILTER (to avoid noise):
- Do NOT extract generic boilerplate or vague caution as "Risk Disclosure" unless it is company-specific and concrete.
- If management quantifies a headwind, states it is worsening/new, or ties it to near-term results, extract it.

BALANCE REQUIREMENT (minimum mix; adjust if transcript is short but keep total >30):
- Forward-Looking: target at least 10 items
- Risk Disclosure: target at least 8 items (specific, not boilerplate)
- Result: target at least 8 items
- Sentiment: at least 2 items
- Macro: at least 2 items
If some types are genuinely not mentioned, you may still create items with Value quoting the closest relevant statement and Reason noting "not explicitly quantified" — but avoid inventing facts.

The transcript is {{transcript_chunk}}

Output as many items as you can find, ideally 30-70. You MUST output more than 30 facts.
Do not include [ORG] in your response.
---

### OUTPUT RULES
* Use the exact markdown block below for **every** extracted item.
* Increment the item number sequentially (1, 2, 3 …).
* One metric per block; never combine multiple metrics.
* Metric should be as specific as possible and include timeframe when available (e.g., "Revenue guidance (next quarter)", "Margin outlook (FY)", "Demand trend (near-term)").
* Value should quote the exact wording/numbers from the transcript.
* Reason should briefly explain why this item could matter for the next ~30 trading days (no predictions, just relevance).

Example output:
### Fact No. 1
- **Type:** <Result | Forward-Looking | Risk Disclosure | Sentiment | Macro>
- **Metric:** Revenue (quarter)
- **Value:** "3 million dollars"
- **Reason:** An achieved revenue result can anchor near-term expectations and post-earnings repricing.""".strip()


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
    print("\nUpdating prompts with BALANCED T+30 versions:")

    prompts["MAIN_AGENT_PROMPT"] = NEW_MAIN_AGENT_PROMPT
    print("  - MAIN_AGENT_PROMPT: UPDATED (balanced net-score approach)")

    prompts["FACTS_EXTRACTION_PROMPT"] = NEW_FACTS_EXTRACTION_PROMPT
    print("  - FACTS_EXTRACTION_PROMPT: UPDATED (T+30 relevance filter, risk noise reduction)")

    # Save as garen1212v2
    print("\nSaving new profile 'garen1212v2'...")
    set_prompt_profile("garen1212v2", prompts)

    print("\n" + "=" * 70)
    print("SUCCESS: Created profile 'garen1212v2'")
    print("=" * 70)
    print(f"\nTotal prompts: {len(prompts)}")
    print("\nKey changes from garen1212 (fixes over-bearish bias):")
    print("  1. MAIN_AGENT_PROMPT")
    print("     - Changed from 'trigger-based MUST DOWN' to 'weighted net-score'")
    print("     - Base score = 5 (neutral), adjust with +2/-2 (strong) or +1/-1 (mild)")
    print("     - Generic risk disclosures NO LONGER auto-trigger bearish")
    print("     - Only company-specific, worsening, near-term risks count as bearish")
    print("     - NEUTRAL (4-6) is default for mixed/vague evidence")
    print("     - Scores 0-1 and 9-10 require very strong unambiguous evidence")
    print("")
    print("  2. FACTS_EXTRACTION_PROMPT")
    print("     - Added RISK DISCLOSURE FILTER to exclude generic boilerplate")
    print("     - Rebalanced minimum quotas:")
    print("       * Forward-Looking >= 10")
    print("       * Risk Disclosure >= 8 (specific only)")
    print("       * Result >= 8")
    print("       * Sentiment >= 2")
    print("       * Macro >= 2")
    print("")
    print("Expected test results with garen1212v2:")
    print("  - Direction distribution should include NEUTRAL (4-6) and UP (7-10)")
    print("  - No longer 100% DOWN predictions")
    print("  - Better balance between GAINER and LOSER hit rates")
    print("")
    print("To apply this profile:")
    print("  POST /api/prompt_profiles/apply with body: {\"name\": \"garen1212v2\"}")
    print("  Or select 'garen1212v2' from the UI dropdown and click Apply")


if __name__ == "__main__":
    main()
