#!/usr/bin/env python3
"""
Batch test for 600 random S&P 500 earnings calls with STRATIFIED sampling.
Ensures each year (2020-2025) is represented proportionally.

Usage:
    python run_batch_600_random_stratified.py garen1212v2
    python run_batch_600_random_stratified.py garen1212v3
    python run_batch_600_random_stratified.py garen1212v2 --test  # Run only 1 for verification
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
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from analysis_engine import analyze_earnings_async
from storage import set_prompt, get_prompt_profile

# Configuration
CONCURRENCY = 15
REPORT_INTERVAL_SECONDS = 120  # 2 minutes
SAMPLE_SIZE = 600  # Total random sample size
YEAR_RANGE = (2020, 2025)  # Fiscal year range
MAX_QUARTER_2025 = 2  # 2025 only up to Q2

# Database connection
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://postgres:password@127.0.0.1:15432/pead_reversal")


def load_test_cases_from_db(sample_size: int, seed: int = 42) -> List[Dict]:
    """Load stratified random test cases from AWS PostgreSQL database.
    Ensures each year has representation proportional to available data.
    """
    conn = psycopg2.connect(POSTGRES_DSN)
    cur = conn.cursor()

    # First, get counts per year
    cur.execute("""
        SELECT
            year,
            COUNT(*) as cnt
        FROM earnings_transcripts
        WHERE year >= %s
          AND year <= %s
          AND (year < 2025 OR (year = 2025 AND quarter <= %s))
          AND t_day IS NOT NULL
        GROUP BY year
        ORDER BY year
    """, (YEAR_RANGE[0], YEAR_RANGE[1], MAX_QUARTER_2025))

    year_counts = dict(cur.fetchall())
    total_available = sum(year_counts.values())

    print(f"\n--- Database Year Distribution ---", flush=True)
    for y in sorted(year_counts.keys()):
        print(f"  {y}: {year_counts[y]:,} records ({year_counts[y]/total_available*100:.1f}%)", flush=True)
    print(f"  Total: {total_available:,}", flush=True)

    # Calculate how many to sample from each year (proportional)
    # But ensure at least some minimum per year (only for large sample sizes)
    num_years = len(year_counts)
    year_samples = {}

    if sample_size < num_years * 10:
        # Small sample: just pick randomly from all years proportionally
        # For test mode (sample_size=1), pick 1 sample from a random year
        for year in year_counts.keys():
            proportion = year_counts[year] / total_available
            year_samples[year] = max(1, int(sample_size * proportion))
        # Ensure we get exactly sample_size total
        total_allocated = sum(year_samples.values())
        while total_allocated > sample_size:
            # Remove from smallest allocation
            for y in sorted(year_samples.keys(), key=lambda x: year_samples[x]):
                if year_samples[y] > 0:
                    year_samples[y] -= 1
                    total_allocated -= 1
                    break
            if total_allocated <= sample_size:
                break
    else:
        # Large sample: ensure minimum per year for stratification
        MIN_PER_YEAR = 50
        for year in year_counts.keys():
            proportion = year_counts[year] / total_available
            year_samples[year] = max(MIN_PER_YEAR, int(sample_size * proportion))

        # Adjust to hit exact sample_size
        total_allocated = sum(year_samples.values())
        if total_allocated != sample_size:
            diff = sample_size - total_allocated
            # Add/remove from the largest year
            largest_year = max(year_counts.keys(), key=lambda y: year_counts[y])
            year_samples[largest_year] += diff

    print(f"\n--- Target Sample Distribution ---", flush=True)
    for y in sorted(year_samples.keys()):
        print(f"  {y}: {year_samples[y]} samples", flush=True)
    print(f"  Total: {sum(year_samples.values())}", flush=True)

    # Set seed for reproducibility
    random.seed(seed)

    # Query and sample from each year
    all_samples = []

    for year in sorted(year_samples.keys()):
        target_count = year_samples[year]

        if year == 2025:
            quarter_filter = f"AND quarter <= {MAX_QUARTER_2025}"
        else:
            quarter_filter = ""

        cur.execute(f"""
            SELECT
                symbol,
                year,
                quarter,
                t_day,
                market_timing
            FROM earnings_transcripts
            WHERE year = %s
              AND t_day IS NOT NULL
              {quarter_filter}
            ORDER BY RANDOM()
            LIMIT %s
        """, (year, target_count))

        records = cur.fetchall()
        print(f"  Sampled {len(records)} from {year}", flush=True)
        all_samples.extend(records)

    cur.close()
    conn.close()

    # Shuffle all samples together
    random.shuffle(all_samples)

    print(f"\nTotal sampled: {len(all_samples)}", flush=True)

    # Convert to test cases
    test_cases = []
    for i, (symbol, year, quarter, t_day, market_timing) in enumerate(all_samples):
        session = "BMO" if market_timing and market_timing.lower() == "bmo" else "AMC"
        test_cases.append({
            "rank": i + 1,
            "symbol": symbol,
            "year": year,
            "quarter": quarter,
            "earnings_date": t_day.strftime("%Y-%m-%d") if t_day else None,
            "session": session,
            "category": "RANDOM",
            "csv_t30_change": None,
            "from_date": None,
            "to_date": None,
            "from_close": None,
            "to_close": None,
        })

    # Print actual distribution
    actual_counts = {}
    for tc in test_cases:
        y = tc["year"]
        actual_counts[y] = actual_counts.get(y, 0) + 1

    print("\n--- Actual Sample Distribution ---", flush=True)
    for y in sorted(actual_counts.keys()):
        print(f"  {y}: {actual_counts[y]} records", flush=True)

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
CURRENT_SAMPLE_SIZE: int = SAMPLE_SIZE


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

                # Extract agentic_result
                agentic_result = analysis_result.get("agentic_result", {})

                # Extract raw output
                raw_data = agentic_result.get("raw", {})
                if isinstance(raw_data, dict):
                    raw_output = raw_data.get("output", "") or raw_data.get("raw_output", "")
                else:
                    raw_output = str(raw_data) if raw_data else ""

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
                    direction, confidence = extract_direction(raw_output)
                    result["predicted"] = direction
                    result["confidence"] = confidence
                    result["confidence_normalized"] = confidence

                # Extract tags
                result["tag_financials"] = extract_tag(raw_output, "Financials")
                result["tag_past"] = extract_tag(raw_output, "Past")
                result["tag_peers"] = extract_tag(raw_output, "Peers")

                # Extract summaries
                result["main_summary"] = agentic_result.get("summary", "")

                notes = raw_data.get("notes", {}) if isinstance(raw_data, dict) else {}
                result["notes_financials"] = notes.get("financials", "") if isinstance(notes, dict) else ""
                result["notes_past"] = notes.get("past", "") if isinstance(notes, dict) else ""
                result["notes_peers"] = notes.get("peers", "") if isinstance(notes, dict) else ""

                # Get T+30 actual change
                t30_change = analysis_result.get("post_earnings_return")
                if t30_change is not None:
                    result["t30_actual"] = t30_change
                    result["hit_result"] = determine_hit(result["predicted"], t30_change)
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
    global results_list, start_time, CURRENT_SAMPLE_SIZE

    async with progress_lock:
        completed = len(results_list)
        if completed == 0:
            return

        success = sum(1 for r in results_list if r["status"] == "success")
        errors = sum(1 for r in results_list if r["status"] != "success")

        valid_results = [r for r in results_list if r["hit_result"] in ["HIT", "MISS"]]
        hits = sum(1 for r in valid_results if r["hit_result"] == "HIT")
        hit_rate = (hits / len(valid_results) * 100) if valid_results else 0

        up_count = sum(1 for r in results_list if r["predicted"] == "UP" and r["status"] == "success")
        down_count = sum(1 for r in results_list if r["predicted"] == "DOWN" and r["status"] == "success")
        neutral_count = sum(1 for r in results_list if r["predicted"] == "NEUTRAL" and r["status"] == "success")

        elapsed = time.time() - start_time
        rate = completed / elapsed * 60 if elapsed > 0 else 0

        print(f"\n{'='*80}", flush=True)
        print(f"PROGRESS REPORT - {datetime.now().strftime('%H:%M:%S')}", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Processed: {completed}/{CURRENT_SAMPLE_SIZE} ({completed/CURRENT_SAMPLE_SIZE*100:.1f}%)", flush=True)
        print(f"Success: {success}, Errors: {errors}", flush=True)
        print(f"Hit Rate: {hit_rate:.1f}% ({hits}/{len(valid_results)})", flush=True)
        print(f"Direction: UP={up_count}, NEUTRAL={neutral_count}, DOWN={down_count}", flush=True)
        print(f"Rate: {rate:.1f} stocks/min", flush=True)
        print(f"Elapsed: {elapsed/60:.1f} min", flush=True)

        remaining = CURRENT_SAMPLE_SIZE - completed
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
    global start_time, results_list, CURRENT_SAMPLE_SIZE

    # Parse arguments
    parser = argparse.ArgumentParser(description='Run stratified batch test with specified profile')
    parser.add_argument('profile', type=str, help='Profile name (e.g., garen1212v2 or garen1212v3)')
    parser.add_argument('--test', action='store_true', help='Run only 1 case for verification')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    profile_name = args.profile
    test_mode = args.test
    seed = args.seed

    if test_mode:
        CURRENT_SAMPLE_SIZE = 1
        sample_size = 1
    else:
        CURRENT_SAMPLE_SIZE = SAMPLE_SIZE
        sample_size = SAMPLE_SIZE

    print("=" * 100, flush=True)
    print(f"STRATIFIED BATCH TEST - {sample_size} Random S&P 500 Earnings Calls", flush=True)
    print(f"Profile: {profile_name}", flush=True)
    print(f"Year Range: {YEAR_RANGE[0]} - {YEAR_RANGE[1]} Q{MAX_QUARTER_2025}", flush=True)
    print(f"Sample Size: {sample_size}", flush=True)
    print(f"Seed: {seed}", flush=True)
    print(f"Test Mode: {test_mode}", flush=True)
    print(f"Concurrency: {CONCURRENCY}", flush=True)
    print(f"Started: {datetime.now().isoformat()}", flush=True)
    print("=" * 100, flush=True)

    # Apply profile
    print(f"\nApplying profile '{profile_name}'...", flush=True)
    if not apply_profile(profile_name):
        print("Failed to apply profile. Exiting.", flush=True)
        return

    # Load test cases
    print("\nLoading test cases from AWS PostgreSQL (stratified sampling)...", flush=True)
    test_cases = load_test_cases_from_db(sample_size, seed)

    if not test_cases:
        print("No test cases found. Exiting.", flush=True)
        return

    print(f"\nLoaded {len(test_cases)} test cases", flush=True)

    if test_mode:
        print("\n" + "=" * 100, flush=True)
        print("TEST MODE: Running single case for verification", flush=True)
        print("=" * 100, flush=True)

    print("\nStarting analysis...", flush=True)

    semaphore = asyncio.Semaphore(CONCURRENCY)

    # Start progress reporter (skip for test mode)
    reporter_task = None
    if not test_mode:
        reporter_task = asyncio.create_task(progress_reporter(REPORT_INTERVAL_SECONDS))

    start_time = time.time()

    # Process all test cases
    tasks = []
    for i, tc in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] {tc['symbol']} {tc['year']}-Q{tc['quarter']}...", flush=True)
        task = asyncio.create_task(process_single(tc, semaphore))
        tasks.append(task)

    # Collect results
    for coro in asyncio.as_completed(tasks):
        result = await coro
        async with progress_lock:
            results_list.append(result)

    # Cancel reporter
    if reporter_task:
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

    if test_mode and len(results_list) > 0:
        # Show detailed output for verification
        r = results_list[0]
        print("\n" + "=" * 100, flush=True)
        print("VERIFICATION OUTPUT", flush=True)
        print("=" * 100, flush=True)
        print(f"Symbol: {r['symbol']}", flush=True)
        print(f"Year/Quarter: {r['year']}-Q{r['quarter']}", flush=True)
        print(f"Status: {r['status']}", flush=True)
        print(f"Predicted: {r['predicted']}", flush=True)
        print(f"Confidence: {r['confidence']}", flush=True)
        print(f"T+30 Actual: {r['t30_actual']}", flush=True)
        print(f"Hit Result: {r['hit_result']}", flush=True)
        print(f"Elapsed: {r['elapsed_seconds']}s", flush=True)
        print(f"Error: {r['error']}", flush=True)
        print(f"\nMain Summary: {r['main_summary'][:500]}..." if r['main_summary'] else "Main Summary: (empty)", flush=True)
        print(f"\nRaw Output (first 1000 chars):", flush=True)
        print(r['raw_output'][:1000] if r['raw_output'] else "(empty)", flush=True)
        return

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
    if success_results:
        print(f"UP (7-10): {up_count} ({up_count/len(success_results)*100:.1f}%)", flush=True)
        print(f"NEUTRAL (4-6): {neutral_count} ({neutral_count/len(success_results)*100:.1f}%)", flush=True)
        print(f"DOWN (0-3): {down_count} ({down_count/len(success_results)*100:.1f}%)", flush=True)

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

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(__file__).resolve().parent / f"batch_600_stratified_{profile_name}_{timestamp}.csv"

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

    print(f"\nCSV saved to: {csv_path}", flush=True)

    json_path = csv_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, indent=2, default=str)

    print(f"JSON saved to: {json_path}", flush=True)
    print("\n" + "=" * 100, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
