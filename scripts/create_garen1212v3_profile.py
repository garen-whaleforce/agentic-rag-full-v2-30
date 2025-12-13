#!/usr/bin/env python3
"""
Create garen1212v3 profile with improved T+30 prompts.

Key changes from v2:
- Evidence-gated scoring: requires hard evidence to move away from NEUTRAL (5)
- Vulnerability cap: limits UP scores for weak fundamentals
- Risk tagging in FACTS_EXTRACTION for proper severity assessment
- Forward-looking tagging for guidance direction
- Stronger anti-false-UP logic for LOSERS
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from storage import get_prompt_profile, set_prompt_profile


# ==============================================================================
# NEW_MAIN_AGENT_PROMPT v3 - Evidence-gated scoring with vulnerability cap
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

EVIDENCE-GATED SCORING RUBRIC (prevents both false UP and false DOWN):

BASE SCORE = 5 (NEUTRAL). You MUST start here and require HARD EVIDENCE to move away.
- "Hard evidence" means: explicit numbers, quantified guidance changes, clear demand/margin statements with metrics, concrete actions (buybacks/capex amounts), or near-term catalysts with dates.
- "Soft evidence" (tone, sentiment, vague optimism, boilerplate) does NOT count as hard evidence.

SCORE GATES (you MUST satisfy these requirements):

Score 6: Requires at least 1 hard-evidence 30D bullish driver AND no high-severity company-specific near-term bearish driver.
         If evidence is merely "tone positive" or "reiterated guidance" without quantified improvement, stay at 5.

Score 7-8: Requires at least 2 independent hard-evidence 30D bullish drivers AND no high-severity company-specific near-term risk.
           Pure sentiment/tone cannot push into this range. Both drivers must have specific evidence.

Score 9-10: RARE. Requires multiple hard-evidence bullish drivers with near-term catalysts AND virtually no material risks.
            Reserve for exceptional cases with clear, quantified acceleration.

Score 4: Only if there is a mild/uncertain negative driver. If mixed/unclear, stay at 5.

Score 0-3: Requires EITHER (a) at least 1 high-severity + company-specific + near-term bearish driver
           (e.g., guidance cut, demand deterioration with numbers, margin compression with drivers, liquidity stress, major contract loss, regulatory action)
           OR (b) at least 2 independent medium-severity bearish drivers.

VULNERABILITY CAP (to reduce false UP on weak fundamentals):
If the financials/notes show vulnerability (e.g., YoY bottom-line deterioration, cash flow pressure, high leverage, cyclical exposure without clear positive guidance), AND there is no raised guidance or explicit demand acceleration, then CAP the score at 5 (maximum 6).
Do NOT use external knowledge; rely only on transcript + notes + financials section.

RISK DISCLOSURE RULES:
- Generic/boilerplate risk language does NOT count as a bearish driver.
- Only count a risk if it has tags: [Specificity=company_specific] [Timeframe=near_term] [Severity=medium or high].
- If a Risk Disclosure lacks these tags or is generic, ignore it for scoring purposes.

SENTIMENT AS AUXILIARY ONLY:
- Sentiment/tone can support a score but CANNOT alone push into 7+ or 0-3.
- If only positive tone exists without hard evidence, stay at 5.
- If only negative tone exists without hard evidence, stay at 5.

Respond in **exactly** this format:

<Couple of sentences of Explanation>

**Summary: <Two sentences supporting your verdict with facts and evidence>, Direction : <0-10>**

Summary requirements (still exactly two sentences):
- Sentence 1: the single most important 30D driver (with specific evidence: numbers or explicit wording).
- Sentence 2: the single most important 30D risk/offset (with specific evidence).
- Keep the evidence grounded in transcript / notes. Avoid generic claims like "strong execution" or "positive momentum".

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
# NEW_FACTS_EXTRACTION_PROMPT v3 - With tagging for risk/forward-looking
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
- guidance direction (raised/lowered/reiterated), near-term demand trends, margin drivers, pricing power, cost pressures,
  inventory/channel health, capex changes, buybacks/liquidity, major customer/contract events,
  competition intensity, regulatory/legal actions, and clearly stated near-term catalysts/risks.

RISK DISCLOSURE EXTRACTION RULES:
- Do NOT extract generic boilerplate or vague caution as "Risk Disclosure".
- ONLY extract company-specific, concrete risks that management quantifies, describes as worsening/new, or ties to near-term results.
- For EACH Risk Disclosure, you MUST append these tags at the END of the Reason field:
  [Specificity=generic|company_specific] [Timeframe=near_term|medium|long] [Severity=low|medium|high] [Quantified=yes|no]

FORWARD-LOOKING EXTRACTION RULES:
- For EACH Forward-Looking item, you MUST append these tags at the END of the Reason field:
  [Guidance=raised|reiterated|lowered|none|unclear] [Timeframe=near_term|next_quarter|FY|multi_year] [Quantified=yes|no]

BALANCE REQUIREMENT (minimum mix; total MUST exceed 30):
- Forward-Looking: at least 12 items
- Risk Disclosure: at least 10 items (specific, not boilerplate)
- Result: at least 8 items
- Sentiment: at least 2 items
- Macro: at least 2 items
If some types are genuinely not mentioned, you may still create items with Value quoting the closest relevant statement and Reason noting "not explicitly quantified" — but avoid inventing facts.

The transcript is {{transcript_chunk}}

Output as many items as you can find, ideally 35-70. You MUST output more than 30 facts.
Do not include [ORG] in your response.
---

### OUTPUT RULES
* Use the exact markdown block below for **every** extracted item.
* Increment the item number sequentially (1, 2, 3 …).
* One metric per block; never combine multiple metrics.
* Metric should be as specific as possible and include timeframe when available (e.g., "Revenue guidance (next quarter)", "Margin outlook (FY)", "Demand trend (near-term)").
* Value should quote the exact wording/numbers from the transcript.
* Reason should briefly explain why this item could matter for the next ~30 trading days, then append the required tags.

Example output for Risk Disclosure:
### Fact No. 1
- **Type:** Risk Disclosure
- **Metric:** FX headwind impact (Q4)
- **Value:** "We expect $50 million headwind from currency in Q4"
- **Reason:** Quantified near-term headwind directly impacts margins. [Specificity=company_specific] [Timeframe=near_term] [Severity=medium] [Quantified=yes]

Example output for Forward-Looking:
### Fact No. 2
- **Type:** Forward-Looking
- **Metric:** Revenue guidance (next quarter)
- **Value:** "We are raising Q4 guidance to $2.5 billion from $2.3 billion"
- **Reason:** Raised guidance signals demand strength. [Guidance=raised] [Timeframe=next_quarter] [Quantified=yes]

Example output for Result:
### Fact No. 3
- **Type:** Result
- **Metric:** EPS (Q3)
- **Value:** "$1.25, beating consensus of $1.10"
- **Reason:** Beat on bottom line may anchor positive sentiment post-earnings.""".strip()


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
    print("\nUpdating prompts with v3 improvements:")

    prompts["MAIN_AGENT_PROMPT"] = NEW_MAIN_AGENT_PROMPT
    print("  - MAIN_AGENT_PROMPT: UPDATED (evidence-gated scoring, vulnerability cap)")

    prompts["FACTS_EXTRACTION_PROMPT"] = NEW_FACTS_EXTRACTION_PROMPT
    print("  - FACTS_EXTRACTION_PROMPT: UPDATED (risk/forward-looking tagging)")

    # Save as garen1212v3
    print("\nSaving new profile 'garen1212v3'...")
    set_prompt_profile("garen1212v3", prompts)

    print("\n" + "=" * 70)
    print("SUCCESS: Created profile 'garen1212v3'")
    print("=" * 70)
    print(f"\nTotal prompts: {len(prompts)}")
    print("\nKey changes from v2 (fixes false UP on LOSERS):")
    print("  1. MAIN_AGENT_PROMPT")
    print("     - Evidence-gated scoring: hard evidence required to leave NEUTRAL")
    print("     - Score 6: needs 1 hard evidence + no high-severity risk")
    print("     - Score 7-8: needs 2 independent hard evidence drivers")
    print("     - Score 9-10: RARE, exceptional cases only")
    print("     - Vulnerability cap: weak fundamentals cap at 5-6")
    print("     - Sentiment as auxiliary only, cannot push 7+ or 0-3 alone")
    print("")
    print("  2. FACTS_EXTRACTION_PROMPT")
    print("     - Risk tagging: [Specificity] [Timeframe] [Severity] [Quantified]")
    print("     - Forward-looking tagging: [Guidance] [Timeframe] [Quantified]")
    print("     - Higher quotas: FL>=12, Risk>=10, Result>=8")
    print("     - Stricter: only company-specific, concrete risks extracted")
    print("")
    print("Expected test results with garen1212v3:")
    print("  - Fewer false UP predictions on LOSERS")
    print("  - More NEUTRAL scores for weak/mixed evidence")
    print("  - Better LOSER hit rate while maintaining GAINER hit rate")
    print("")
    print("To apply this profile:")
    print("  POST /api/prompt_profiles/apply with body: {\"name\": \"garen1212v3\"}")
    print("  Or select 'garen1212v3' from the UI dropdown and click Apply")


if __name__ == "__main__":
    main()
