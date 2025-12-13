#!/usr/bin/env python3
"""
Batch test for 500 random S&P 500 earnings calls using garen1212v2 profile.
Randomly selects from 2020-2025 Q2 data from AWS PostgreSQL database.

garen1212v2 fixes:
- Changed from 'trigger-based MUST DOWN' to 'weighted net-score'
- Generic risk disclosures no longer auto-trigger bearish
- NEUTRAL (4-6) is default for mixed/vague evidence
"""

import sys
import os
os.environ['PYTHONUNBUFFERED'] = '1'

import asyncio
import csv
import json
import time
import random
import psycopg2
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from analysis_engine import analyze_earnings_async
from storage import set_prompt, get_prompt_profile

# Configuration
PROFILE_NAME = "garen1212v3"  # Evidence-gated T+30 profile (v3)
CONCURRENCY = 15
REPORT_INTERVAL_SECONDS = 300  # 5 minutes for larger test
SAMPLE_SIZE = 500  # Random sample size
YEAR_RANGE = (2020, 2025)  # Fiscal year range
MAX_QUARTER_2025 = 2  # 2025 only up to Q2

# Database connection
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://postgres:password@127.0.0.1:15432/pead_reversal")


def load_test_cases_from_db() -> List[Dict]:
    """Load random test cases from AWS PostgreSQL database."""
    conn = psycopg2.connect(POSTGRES_DSN)
    cur = conn.cursor()

    # Query all eligible transcripts (2020-2025 Q2)
    cur.execute("""
        SELECT
            symbol,
            year,
            quarter,
            t_day,
            market_timing
        FROM earnings_transcripts
        WHERE year >= %s
          AND year <= %s
          AND (year < 2025 OR (year = 2025 AND quarter <= %s))
          AND t_day IS NOT NULL
        ORDER BY year, quarter, symbol
    """, (YEAR_RANGE[0], YEAR_RANGE[1], MAX_QUARTER_2025))

    all_records = cur.fetchall()
    cur.close()
    conn.close()

    print(f"Found {len(all_records):,} eligible records in database", flush=True)

    # Randomly sample
    if len(all_records) <= SAMPLE_SIZE:
        selected = all_records
    else:
        selected = random.sample(all_records, SAMPLE_SIZE)

    print(f"Randomly selected {len(selected)} records", flush=True)

    # Convert to test cases
    test_cases = []
    for i, (symbol, year, quarter, t_day, market_timing) in enumerate(selected):
        session = "BMO" if market_timing and market_timing.lower() == "bmo" else "AMC"
        test_cases.append({
            "rank": i + 1,
            "symbol": symbol,
            "year": year,
            "quarter": quarter,
            "earnings_date": t_day.strftime("%Y-%m-%d") if t_day else None,
            "session": session,
            "category": "RANDOM",  # Not gainer/loser, random sample
            "csv_t30_change": None,  # Will be calculated later
            "from_date": None,
            "to_date": None,
            "from_close": None,
            "to_close": None,
        })

    # Print year distribution
    year_counts = {}
    for tc in test_cases:
        y = tc["year"]
        year_counts[y] = year_counts.get(y, 0) + 1

    print("\n--- Sample Year Distribution ---", flush=True)
    for y in sorted(year_counts.keys()):
        print(f"  {y}: {year_counts[y]} records", flush=True)

    return test_cases


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


import re

def extract_direction(raw_output: str) -> tuple:
    """Extract direction prediction and confidence from raw output."""
    if not raw_output:
        return "N/A", 0.0

    # Look for "Direction : X" pattern (with score 0-10)
    pattern = r'Direction\s*:\s*(\d+(?:\.\d+)?)'
    match = re.search(pattern, raw_output, re.IGNORECASE)

    if match:
        score = float(match.group(1))
        # Normalize score to 0-10 range if needed
        if score <= 1.0:
            score = score * 10

        # Map score to direction
        if score <= 3:
            direction = "DOWN"
        elif score >= 7:
            direction = "UP"
        else:
            direction = "NEUTRAL"

        return direction, score

    return "N/A", 0.0


def extract_tag(raw_output: str, tag_name: str) -> str:
    """Extract tag value from raw output."""
    if not raw_output:
        return "N/A"
    pattern = rf'{tag_name}:\s*(\w+)'
    match = re.search(pattern, raw_output, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return "N/A"


# Global tracking
progress_lock = asyncio.Lock()
results_list: List[Dict] = []
start_time: float = 0
last_report_time: float = 0


async def process_single(test_case: Dict, semaphore: asyncio.Semaphore) -> Dict:
    """Process a single test case."""
    async with semaphore:
        symbol = test_case["symbol"]
        year = test_case["year"]
        quarter = test_case["quarter"]
        category = test_case["category"]

        result = {
            **test_case,
            "predicted": "N/A",
            "confidence": 0.0,
            "confidence_normalized": 0.0,
            "hit_result": "ERROR",
            "status": "pending",
            "elapsed_seconds": 0,
            "tag_financials": "N/A",
            "tag_past": "N/A",
            "tag_peers": "N/A",
            "main_summary": "",
            "notes_financials": "",
            "notes_past": "",
            "notes_peers": "",
            "raw_output": "",
            "error": "",
            "t30_actual": None,
        }

        start = time.time()

        try:
            analysis_result = await analyze_earnings_async(
                symbol=symbol,
                year=year,
                quarter=quarter
            )

            elapsed = time.time() - start
            result["elapsed_seconds"] = round(elapsed, 1)

            if analysis_result.get("error"):
                result["status"] = "error"
                result["error"] = analysis_result.get("error", "Unknown error")
            else:
                result["status"] = "success"

                # Extract agentic_result (contains the actual analysis output)
                agentic_result = analysis_result.get("agentic_result", {})

                # Extract raw output from agentic_result['raw'] (dict) or fall back to 'summary'
                raw_data = agentic_result.get("raw", {})
                if isinstance(raw_data, dict):
                    raw_output = raw_data.get("output", "") or raw_data.get("raw_output", "")
                else:
                    raw_output = str(raw_data) if raw_data else ""

                # Also try summary if raw_output is empty
                if not raw_output:
                    raw_output = agentic_result.get("summary", "")

                result["raw_output"] = raw_output

                # Use prediction/confidence directly from agentic_result if available
                pred = agentic_result.get("prediction", "")
                conf = agentic_result.get("confidence", 0.0)

                if pred:
                    result["predicted"] = pred.upper() if isinstance(pred, str) else "N/A"
                    result["confidence"] = conf if isinstance(conf, (int, float)) else 0.0
                    result["confidence_normalized"] = result["confidence"]
                else:
                    # Fall back to extracting from raw_output
                    direction, confidence = extract_direction(raw_output)
                    result["predicted"] = direction
                    result["confidence"] = confidence
                    result["confidence_normalized"] = confidence

                # Extract tags from raw_output
                result["tag_financials"] = extract_tag(raw_output, "Financials")
                result["tag_past"] = extract_tag(raw_output, "Past")
                result["tag_peers"] = extract_tag(raw_output, "Peers")

                # Extract summaries from agentic_result
                result["main_summary"] = agentic_result.get("summary", "")

                # Notes are in the raw dict under 'notes'
                notes = raw_data.get("notes", {}) if isinstance(raw_data, dict) else {}
                result["notes_financials"] = notes.get("financials", "") if isinstance(notes, dict) else ""
                result["notes_past"] = notes.get("past", "") if isinstance(notes, dict) else ""
                result["notes_peers"] = notes.get("peers", "") if isinstance(notes, dict) else ""

                # Get T+30 actual change from analysis result
                t30_change = analysis_result.get("t30_change")
                if t30_change is not None:
                    result["t30_actual"] = t30_change
                    result["hit_result"] = determine_hit(direction, t30_change)
                else:
                    result["hit_result"] = "NO_DATA"

        except Exception as e:
            elapsed = time.time() - start
            result["elapsed_seconds"] = round(elapsed, 1)
            result["status"] = "exception"
            result["error"] = str(e)

        return result


async def report_progress():
    """Report current progress."""
    global results_list, start_time, last_report_time

    async with progress_lock:
        completed = len(results_list)
        if completed == 0:
            return

        success = sum(1 for r in results_list if r["status"] == "success")
        errors = sum(1 for r in results_list if r["status"] != "success")

        # Calculate hit rate (excluding SKIP and ERROR)
        valid_results = [r for r in results_list if r["hit_result"] in ["HIT", "MISS"]]
        hits = sum(1 for r in valid_results if r["hit_result"] == "HIT")
        hit_rate = (hits / len(valid_results) * 100) if valid_results else 0

        # Direction distribution
        up_count = sum(1 for r in results_list if r["predicted"] == "UP" and r["status"] == "success")
        down_count = sum(1 for r in results_list if r["predicted"] == "DOWN" and r["status"] == "success")
        neutral_count = sum(1 for r in results_list if r["predicted"] == "NEUTRAL" and r["status"] == "success")

        # Average confidence
        confidences = [r["confidence_normalized"] for r in results_list if r["status"] == "success" and r["confidence_normalized"] > 0]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        elapsed = time.time() - start_time
        rate = completed / elapsed * 60 if elapsed > 0 else 0

        print(f"\n{'='*80}", flush=True)
        print(f"PROGRESS REPORT - {datetime.now().strftime('%H:%M:%S')}", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Processed: {completed}/{SAMPLE_SIZE} ({completed/SAMPLE_SIZE*100:.1f}%)", flush=True)
        print(f"Success: {success}, Errors: {errors}", flush=True)
        print(f"Hit Rate: {hit_rate:.1f}% ({hits}/{len(valid_results)})", flush=True)
        print(f"Direction: UP={up_count}, NEUTRAL={neutral_count}, DOWN={down_count}", flush=True)
        print(f"Avg Score: {avg_conf:.2f}", flush=True)
        print(f"Rate: {rate:.1f} stocks/min", flush=True)
        print(f"Elapsed: {elapsed/60:.1f} min", flush=True)

        remaining = SAMPLE_SIZE - completed
        if rate > 0:
            eta_min = remaining / rate
            print(f"ETA: {eta_min:.1f} min remaining", flush=True)

        print(f"{'='*80}\n", flush=True)


async def progress_reporter(interval: int):
    """Periodically report progress."""
    while True:
        await asyncio.sleep(interval)
        await report_progress()


async def main():
    global start_time, last_report_time, results_list

    print("=" * 100, flush=True)
    print(f"BATCH TEST - {SAMPLE_SIZE} Random S&P 500 Earnings Calls with garen1212v2 Profile", flush=True)
    print(f"Profile: {PROFILE_NAME}", flush=True)
    print(f"Year Range: {YEAR_RANGE[0]} - {YEAR_RANGE[1]} Q{MAX_QUARTER_2025}", flush=True)
    print(f"Sample Size: {SAMPLE_SIZE}", flush=True)
    print(f"Concurrency: {CONCURRENCY}", flush=True)
    print(f"Report Interval: {REPORT_INTERVAL_SECONDS}s ({REPORT_INTERVAL_SECONDS//60} min)", flush=True)
    print(f"Started: {datetime.now().isoformat()}", flush=True)
    print("=" * 100, flush=True)

    # Apply profile
    print(f"\nApplying profile '{PROFILE_NAME}'...", flush=True)
    if not apply_profile(PROFILE_NAME):
        print("Failed to apply profile. Exiting.", flush=True)
        return

    # Load test cases from database
    print("\nLoading test cases from AWS PostgreSQL...", flush=True)
    test_cases = load_test_cases_from_db()

    if not test_cases:
        print("No test cases found. Exiting.", flush=True)
        return

    print(f"\nLoaded {len(test_cases)} test cases", flush=True)

    print("\n" + "=" * 100, flush=True)
    print("Starting analysis...", flush=True)

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(CONCURRENCY)

    # Start progress reporter
    reporter_task = asyncio.create_task(progress_reporter(REPORT_INTERVAL_SECONDS))

    start_time = time.time()
    last_report_time = start_time

    # Process all test cases
    tasks = []
    for i, tc in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] {tc['symbol']} {tc['year']}-Q{tc['quarter']}...", flush=True)
        task = asyncio.create_task(process_single(tc, semaphore))
        tasks.append(task)

    # Collect results as they complete
    for coro in asyncio.as_completed(tasks):
        result = await coro
        async with progress_lock:
            results_list.append(result)

    # Cancel reporter
    reporter_task.cancel()
    try:
        await reporter_task
    except asyncio.CancelledError:
        pass

    # Final report
    total_time = time.time() - start_time

    print("\n" + "=" * 100, flush=True)
    print("BATCH TEST COMPLETE", flush=True)
    print("=" * 100, flush=True)

    success_results = [r for r in results_list if r["status"] == "success"]
    error_results = [r for r in results_list if r["status"] != "success"]

    print(f"\nTotal processed: {len(results_list)}", flush=True)
    print(f"Success: {len(success_results)}", flush=True)
    print(f"Errors: {len(error_results)}", flush=True)
    print(f"Total time: {total_time/60:.1f} minutes", flush=True)
    print(f"Rate: {len(results_list)/total_time*60:.1f} stocks/min", flush=True)

    # Hit rate analysis
    valid_results = [r for r in success_results if r["hit_result"] in ["HIT", "MISS"]]
    hits = sum(1 for r in valid_results if r["hit_result"] == "HIT")

    print(f"\n--- HIT RATE ANALYSIS ---", flush=True)
    print(f"Valid predictions: {len(valid_results)}", flush=True)
    print(f"Hits: {hits}", flush=True)
    print(f"Overall Hit Rate: {hits/len(valid_results)*100:.1f}%" if valid_results else "N/A", flush=True)

    # Direction distribution
    up_count = sum(1 for r in success_results if r["predicted"] == "UP")
    down_count = sum(1 for r in success_results if r["predicted"] == "DOWN")
    neutral_count = sum(1 for r in success_results if r["predicted"] == "NEUTRAL")

    print(f"\n--- DIRECTION DISTRIBUTION ---", flush=True)
    print(f"UP (7-10): {up_count} ({up_count/len(success_results)*100:.1f}%)" if success_results else "N/A", flush=True)
    print(f"NEUTRAL (4-6): {neutral_count} ({neutral_count/len(success_results)*100:.1f}%)" if success_results else "N/A", flush=True)
    print(f"DOWN (0-3): {down_count} ({down_count/len(success_results)*100:.1f}%)" if success_results else "N/A", flush=True)

    # Score distribution
    scores = [r["confidence_normalized"] for r in success_results if r["confidence_normalized"] > 0]
    if scores:
        print(f"\n--- SCORE DISTRIBUTION ---", flush=True)
        print(f"Mean: {sum(scores)/len(scores):.2f}", flush=True)
        print(f"Min: {min(scores):.1f}", flush=True)
        print(f"Max: {max(scores):.1f}", flush=True)

    # Year breakdown
    print(f"\n--- HIT RATE BY YEAR ---", flush=True)
    year_results = {}
    for r in valid_results:
        y = r["year"]
        if y not in year_results:
            year_results[y] = {"hits": 0, "total": 0}
        year_results[y]["total"] += 1
        if r["hit_result"] == "HIT":
            year_results[y]["hits"] += 1

    for y in sorted(year_results.keys()):
        yr = year_results[y]
        rate = yr["hits"] / yr["total"] * 100 if yr["total"] > 0 else 0
        print(f"  {y}: {yr['hits']}/{yr['total']} = {rate:.1f}%", flush=True)

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(__file__).resolve().parent / f"batch_500_random_sp500_{timestamp}.csv"

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            "rank", "symbol", "year", "quarter", "earnings_date", "category", "session",
            "predicted", "confidence", "confidence_normalized", "csv_t30_change", "t30_actual", "hit_result",
            "status", "elapsed_seconds", "from_date", "to_date", "from_close", "to_close",
            "tag_financials", "tag_past", "tag_peers",
            "main_summary", "notes_financials", "notes_past", "notes_peers",
            "raw_output", "error"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in sorted(results_list, key=lambda x: x["rank"]):
            writer.writerow(r)

    print(f"\nResults saved to: {csv_path}", flush=True)

    # Also save as JSON for detailed analysis
    json_path = csv_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, indent=2, default=str)

    print(f"JSON saved to: {json_path}", flush=True)
    print("\n" + "=" * 100, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
