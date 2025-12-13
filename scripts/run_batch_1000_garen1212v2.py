#!/usr/bin/env python3
"""
Batch test for top 1000 earnings moves (500 gainers + 500 losers) using garen1212v2 profile.
Uses the provided CSVs with pre-calculated T+30 data.

garen1212v2 fixes:
- Changed from 'trigger-based MUST DOWN' to 'weighted net-score'
- Generic risk disclosures no longer auto-trigger bearish
- NEUTRAL (4-6) is default for mixed/vague evidence

Features:
- Applies garen1212v2 profile (balanced T+30 prompts)
- Skip cache for fresh analysis
- Summary report every 5 minutes
- Final CSV with all agent detailed outputs
"""

import sys
import os
os.environ['PYTHONUNBUFFERED'] = '1'

import asyncio
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from analysis_engine import analyze_earnings_async
from storage import set_prompt, get_prompt_profile

# Configuration
PROFILE_NAME = "garen1212v2"  # Balanced T+30 profile
CONCURRENCY = 15
REPORT_INTERVAL_SECONDS = 300  # 5 minutes
GAINERS_CSV = Path(__file__).resolve().parent / "top500_gainers_2020_2025.csv"
LOSERS_CSV = Path(__file__).resolve().parent / "top500_losers_2020_2025.csv"


def parse_pct(pct_str: str) -> float:
    """Parse percentage string like '134.62%' to float 134.62"""
    if not pct_str:
        return 0.0
    return float(pct_str.replace('%', '').strip())


def determine_fiscal_quarter(earnings_date: str) -> tuple:
    """
    Determine fiscal year and quarter from earnings date.
    Earnings typically reported ~1 month after quarter end:
    - Q1 (Jan-Mar) reported in Apr-May
    - Q2 (Apr-Jun) reported in Jul-Aug
    - Q3 (Jul-Sep) reported in Oct-Nov
    - Q4 (Oct-Dec) reported in Jan-Feb next year
    """
    dt = datetime.strptime(earnings_date, "%Y-%m-%d")
    month = dt.month
    year = dt.year

    if month in [1, 2]:  # Q4 of previous year
        return year - 1, 4
    elif month in [4, 5]:  # Q1
        return year, 1
    elif month in [7, 8]:  # Q2
        return year, 2
    elif month in [10, 11]:  # Q3
        return year, 3
    elif month == 3:  # Could be Q4 (late) or Q1 (early)
        return year - 1, 4
    elif month == 6:  # Q1 (late) or Q2 (early)
        return year, 1
    elif month == 9:  # Q2 (late) or Q3 (early)
        return year, 2
    elif month == 12:  # Q3 (late) or Q4 (early)
        return year, 3

    return year, (month - 1) // 3 + 1


def determine_hit(prediction: str, actual_change: float) -> str:
    """Determine if prediction was a HIT, MISS, or SKIP"""
    if not prediction or prediction == "N/A":
        return "ERROR"

    pred_upper = prediction.upper()
    if pred_upper == "NEUTRAL":
        return "SKIP"

    actual_direction = "UP" if actual_change > 0 else "DOWN"

    if pred_upper == actual_direction:
        return "HIT"
    else:
        return "MISS"


def apply_profile(profile_name: str) -> bool:
    """Apply a saved profile to current settings."""
    profile = get_prompt_profile(profile_name)
    if not profile:
        print(f"ERROR: Profile '{profile_name}' not found!", flush=True)
        return False
    prompts = profile.get("prompts", {})
    for key, content in prompts.items():
        set_prompt(key, content)
    print(f"Applied profile '{profile_name}' with {len(prompts)} prompts", flush=True)
    return True


def extract_short_term_tag(note_text: str) -> str:
    """Extract ShortTermTag from helper note."""
    if not note_text:
        return "N/A"
    import re
    match = re.search(r"ShortTermTag:\s*(Bullish|Bearish|Neutral|Unclear)", note_text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return "N/A"


def load_test_cases() -> List[Dict]:
    """Load test cases from both CSV files"""
    test_cases = []

    # Load gainers
    with open(GAINERS_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rank = int(row['rank'])
            symbol = row['symbol']
            earnings_date = row['earnings_date']
            pct_change = parse_pct(row['pct_change'])
            session = "BMO" if row['time_of_day'] == 'bmo' else "AMC"

            fiscal_year, fiscal_quarter = determine_fiscal_quarter(earnings_date)

            test_cases.append({
                "rank": rank,
                "symbol": symbol,
                "year": fiscal_year,
                "quarter": fiscal_quarter,
                "earnings_date": earnings_date,
                "session": session,
                "category": "GAINER",
                "csv_t30_change": pct_change,
                "from_date": row['from_date'],
                "to_date": row['to_date'],
                "from_close": float(row['from_close']) if row['from_close'] else None,
                "to_close": float(row['to_close']) if row['to_close'] else None,
            })

    # Load losers
    with open(LOSERS_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rank = int(row['rank'])
            symbol = row['symbol']
            earnings_date = row['earnings_date']
            pct_change = parse_pct(row['pct_change'])
            session = "BMO" if row['time_of_day'] == 'bmo' else "AMC"

            fiscal_year, fiscal_quarter = determine_fiscal_quarter(earnings_date)

            test_cases.append({
                "rank": rank + 500,  # Offset rank for losers
                "symbol": symbol,
                "year": fiscal_year,
                "quarter": fiscal_quarter,
                "earnings_date": earnings_date,
                "session": session,
                "category": "LOSER",
                "csv_t30_change": pct_change,
                "from_date": row['from_date'],
                "to_date": row['to_date'],
                "from_close": float(row['from_close']) if row['from_close'] else None,
                "to_close": float(row['to_close']) if row['to_close'] else None,
            })

    return test_cases


# Global tracking
progress_lock = asyncio.Lock()
results_list: List[Dict] = []
start_time: float = 0
last_report_time: float = 0


def print_summary_report(results: List[Dict], elapsed_seconds: float, total: int, is_final: bool = False):
    """Print a summary report."""
    report_type = "FINAL SUMMARY" if is_final else "PROGRESS REPORT"
    print(f"\n{'='*100}", flush=True)
    print(f"{report_type} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Profile: {PROFILE_NAME}", flush=True)
    print(f"Elapsed: {elapsed_seconds/60:.1f} minutes | Processed: {len(results)}/{total}", flush=True)
    print(f"{'='*100}", flush=True)

    success = [r for r in results if r.get("status") == "success"]
    errors = [r for r in results if r.get("status") != "success"]

    print(f"\nSuccess: {len(success)} | Errors: {len(errors)}", flush=True)

    if success:
        hits = [r for r in success if r.get("hit_result") == "HIT"]
        misses = [r for r in success if r.get("hit_result") == "MISS"]
        skips = [r for r in success if r.get("hit_result") == "SKIP"]

        print(f"\nOverall: HIT={len(hits)} | MISS={len(misses)} | SKIP={len(skips)}", flush=True)
        trades = len(hits) + len(misses)
        if trades > 0:
            hit_rate = len(hits) / trades * 100
            print(f"Hit Rate (HIT/(HIT+MISS)): {hit_rate:.1f}% ({len(hits)}/{trades})", flush=True)

        # Prediction distribution
        up_pred = [r for r in success if r.get("predicted") == "UP"]
        down_pred = [r for r in success if r.get("predicted") == "DOWN"]
        neutral_pred = [r for r in success if r.get("predicted") == "NEUTRAL"]
        print(f"Prediction Distribution: UP={len(up_pred)}, DOWN={len(down_pred)}, NEUTRAL={len(neutral_pred)}", flush=True)

        # Category breakdown
        gainer_results = [r for r in success if r.get("category") == "GAINER"]
        loser_results = [r for r in success if r.get("category") == "LOSER"]

        if gainer_results:
            g_hits = sum(1 for r in gainer_results if r.get("hit_result") == "HIT")
            g_misses = sum(1 for r in gainer_results if r.get("hit_result") == "MISS")
            g_skips = sum(1 for r in gainer_results if r.get("hit_result") == "SKIP")
            g_trades = g_hits + g_misses
            g_rate = g_hits / g_trades * 100 if g_trades > 0 else 0
            print(f"\nGAINERS ({len(gainer_results)}): HIT={g_hits} MISS={g_misses} SKIP={g_skips} Rate={g_rate:.1f}%", flush=True)

        if loser_results:
            l_hits = sum(1 for r in loser_results if r.get("hit_result") == "HIT")
            l_misses = sum(1 for r in loser_results if r.get("hit_result") == "MISS")
            l_skips = sum(1 for r in loser_results if r.get("hit_result") == "SKIP")
            l_trades = l_hits + l_misses
            l_rate = l_hits / l_trades * 100 if l_trades > 0 else 0
            print(f"LOSERS ({len(loser_results)}): HIT={l_hits} MISS={l_misses} SKIP={l_skips} Rate={l_rate:.1f}%", flush=True)

        # Direction score distribution - KEY METRIC for v2 validation
        scores = [r.get("confidence") for r in success if r.get("confidence") is not None]
        if scores:
            avg_score = sum(scores) / len(scores)
            low_scores = sum(1 for s in scores if s <= 3)
            mid_scores = sum(1 for s in scores if 4 <= s <= 6)
            high_scores = sum(1 for s in scores if s >= 7)
            print(f"\nDirection Scores: Avg={avg_score:.2f} | 0-3={low_scores} ({low_scores/len(scores)*100:.1f}%) | 4-6={mid_scores} ({mid_scores/len(scores)*100:.1f}%) | 7-10={high_scores} ({high_scores/len(scores)*100:.1f}%)", flush=True)

            # Additional validation check
            if mid_scores == 0 and high_scores == 0:
                print("  ⚠️  WARNING: No NEUTRAL or UP predictions - prompt may still be too bearish!", flush=True)
            elif low_scores == 0 and mid_scores == 0:
                print("  ⚠️  WARNING: No DOWN or NEUTRAL predictions - prompt may be too bullish!", flush=True)
            else:
                print("  ✓ Distribution includes multiple score ranges - prompt appears balanced", flush=True)

    print(f"{'='*100}\n", flush=True)


async def report_progress_periodically(total: int):
    """Background task to report progress."""
    global last_report_time, results_list, start_time
    while True:
        await asyncio.sleep(60)
        current_time = time.time()
        if current_time - last_report_time >= REPORT_INTERVAL_SECONDS:
            async with progress_lock:
                if results_list:
                    elapsed = current_time - start_time
                    print_summary_report(results_list.copy(), elapsed, total, is_final=False)
                    last_report_time = current_time


async def analyze_single(test_case: Dict, idx: int, total: int, semaphore: asyncio.Semaphore):
    """Analyze a single test case"""
    global results_list

    async with semaphore:
        rank = test_case["rank"]
        symbol = test_case["symbol"]
        year = test_case["year"]
        quarter = test_case["quarter"]
        category = test_case["category"]
        t30_change = test_case["csv_t30_change"]

        result_row = {
            **test_case,
            "predicted": None,
            "confidence": None,
            "hit_result": None,
            "status": "pending",
            "error": None,
            "elapsed_seconds": None,
            # Detailed agent outputs
            "main_summary": None,
            "notes_financials": None,
            "notes_past": None,
            "notes_peers": None,
            "tag_financials": None,
            "tag_past": None,
            "tag_peers": None,
            "raw_output": None,
        }

        try:
            print(f"[{idx}/{total}] {symbol} {year}-Q{quarter} ({category})...", flush=True)

            start = time.time()
            analysis_result = await analyze_earnings_async(
                symbol=symbol,
                year=year,
                quarter=quarter,
                skip_cache=True,
            )
            elapsed = time.time() - start
            result_row["elapsed_seconds"] = round(elapsed, 1)

            # Check if analysis succeeded
            if analysis_result and "error" not in analysis_result:
                agentic_result = analysis_result.get("agentic_result", {})
                if isinstance(agentic_result, dict) and agentic_result.get("prediction"):
                    result_row["predicted"] = agentic_result.get("prediction", "N/A")
                    result_row["confidence"] = agentic_result.get("confidence")
                    result_row["main_summary"] = agentic_result.get("summary", "")
                    result_row["hit_result"] = determine_hit(result_row["predicted"], t30_change)
                    result_row["status"] = "success"

                    # Extract detailed notes
                    raw = agentic_result.get("raw", {})
                    if isinstance(raw, dict):
                        notes = raw.get("notes", {})
                        if isinstance(notes, dict):
                            result_row["notes_financials"] = notes.get("financials", "")
                            result_row["notes_past"] = notes.get("past", "")
                            result_row["notes_peers"] = notes.get("peers", "")
                            result_row["tag_financials"] = extract_short_term_tag(notes.get("financials", ""))
                            result_row["tag_past"] = extract_short_term_tag(notes.get("past", ""))
                            result_row["tag_peers"] = extract_short_term_tag(notes.get("peers", ""))
                        result_row["raw_output"] = raw.get("raw_agent_output", "")

                    tags_str = f"[F:{result_row['tag_financials']} P:{result_row['tag_past']} C:{result_row['tag_peers']}]"
                    print(f"  {symbol}: {result_row['predicted']} (conf={result_row['confidence']}) {tags_str} T+30={t30_change:+.1f}% -> {result_row['hit_result']} [{elapsed:.0f}s]", flush=True)
                else:
                    result_row["status"] = "error"
                    result_row["error"] = "No prediction returned"
                    print(f"  {symbol}: ERROR - No prediction [{elapsed:.0f}s]", flush=True)
            else:
                error_msg = analysis_result.get("error", "Unknown") if analysis_result else "No result"
                result_row["status"] = "error"
                result_row["error"] = str(error_msg)[:200]
                print(f"  {symbol}: ERROR - {str(error_msg)[:80]} [{elapsed:.0f}s]", flush=True)

        except Exception as e:
            elapsed = time.time() - start if 'start' in dir() else 0
            result_row["status"] = "exception"
            result_row["error"] = str(e)[:200]
            result_row["elapsed_seconds"] = round(elapsed, 1) if elapsed else None
            print(f"  {symbol}: EXCEPTION - {str(e)[:80]}", flush=True)

        async with progress_lock:
            results_list.append(result_row)


def save_results(results: List[Dict], timestamp: str):
    """Save results to CSV and JSON files."""
    # Sort by rank
    results.sort(key=lambda x: x.get("rank", 9999))

    # CSV with all fields
    csv_columns = [
        'rank', 'symbol', 'year', 'quarter', 'earnings_date', 'category', 'session',
        'predicted', 'confidence', 'csv_t30_change', 'hit_result', 'status',
        'elapsed_seconds', 'from_date', 'to_date', 'from_close', 'to_close',
        'tag_financials', 'tag_past', 'tag_peers',
        'main_summary', 'notes_financials', 'notes_past', 'notes_peers',
        'raw_output', 'error'
    ]

    csv_path = Path(__file__).resolve().parent / f"batch_1000_garen1212v2_{timestamp}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction='ignore')
        writer.writeheader()
        for row in results:
            # Truncate very long text fields
            write_row = row.copy()
            for field in ["main_summary", "notes_financials", "notes_past", "notes_peers", "raw_output"]:
                if write_row.get(field) and len(str(write_row[field])) > 10000:
                    write_row[field] = str(write_row[field])[:10000] + "...[truncated]"
            writer.writerow(write_row)
    print(f"\nCSV saved to: {csv_path}", flush=True)

    # JSON with full data
    json_path = Path(__file__).resolve().parent / f"batch_1000_garen1212v2_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"JSON saved to: {json_path}", flush=True)

    return csv_path, json_path


async def main():
    global start_time, last_report_time, results_list

    print("=" * 100, flush=True)
    print("BATCH TEST - Top 1000 Earnings Moves with garen1212v2 Profile (Balanced T+30)", flush=True)
    print(f"Profile: {PROFILE_NAME}", flush=True)
    print(f"Concurrency: {CONCURRENCY}", flush=True)
    print(f"Report Interval: {REPORT_INTERVAL_SECONDS}s ({REPORT_INTERVAL_SECONDS//60} min)", flush=True)
    print(f"Started: {datetime.now().isoformat()}", flush=True)
    print("=" * 100, flush=True)

    # Apply profile
    print(f"\nApplying profile '{PROFILE_NAME}'...", flush=True)
    if not apply_profile(PROFILE_NAME):
        print(f"Failed to apply profile {PROFILE_NAME}, exiting.", flush=True)
        return []

    # Load test cases
    print("\nLoading test cases from CSVs...", flush=True)
    test_cases = load_test_cases()
    total = len(test_cases)
    print(f"  Loaded {total} test cases", flush=True)

    # Show preview
    gainers = [t for t in test_cases if t["category"] == "GAINER"]
    losers = [t for t in test_cases if t["category"] == "LOSER"]
    print(f"  Gainers: {len(gainers)}, Losers: {len(losers)}", flush=True)

    print("\n--- Top 5 Gainers ---")
    for t in gainers[:5]:
        print(f"  {t['rank']}. {t['symbol']} {t['year']}-Q{t['quarter']} ({t['earnings_date']}) T+30={t['csv_t30_change']:+.2f}%")

    print("\n--- Top 5 Losers ---")
    for t in losers[:5]:
        print(f"  {t['rank']}. {t['symbol']} {t['year']}-Q{t['quarter']} ({t['earnings_date']}) T+30={t['csv_t30_change']:+.2f}%")

    print("\n" + "=" * 100)
    print("Starting analysis...")
    print("KEY VALIDATION: Watch for Direction Score distribution - should see 0-3, 4-6, AND 7-10 ranges")
    print()

    # Initialize
    start_time = time.time()
    last_report_time = start_time
    results_list = []

    semaphore = asyncio.Semaphore(CONCURRENCY)
    reporter_task = asyncio.create_task(report_progress_periodically(total))

    # Run all analyses
    tasks = [
        analyze_single(tc, idx, total, semaphore)
        for idx, tc in enumerate(test_cases, 1)
    ]

    try:
        await asyncio.gather(*tasks)
    finally:
        reporter_task.cancel()
        try:
            await reporter_task
        except asyncio.CancelledError:
            pass

    elapsed = time.time() - start_time

    # Final summary
    print_summary_report(results_list, elapsed, total, is_final=True)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path, json_path = save_results(results_list, timestamp)

    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"Average per ticker: {elapsed/total:.1f}s", flush=True)
    print("=" * 100, flush=True)

    return results_list


if __name__ == "__main__":
    asyncio.run(main())
