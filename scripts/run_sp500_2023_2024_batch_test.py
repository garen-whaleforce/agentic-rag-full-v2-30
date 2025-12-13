#!/usr/bin/env python3
"""
Batch Test Script - S&P 500 2023-2024 Quarterly Earnings

Fetches all S&P 500 transcripts for 2023 and 2024 from AWS PostgreSQL database,
runs analysis on each, and outputs results to CSV.

T+30 Calculation:
- BMO (Before Market Open): T+30 price vs T price (earnings day close)
- AMC (After Market Close): T+30 price vs T+1 price (next trading day close)

Total test cases: ~3941 (504 companies x 8 quarters)

Usage:
    python scripts/run_sp500_2023_2024_batch_test.py [--start-from SYMBOL] [--year YEAR] [--quarter Q]
"""

import asyncio
import csv
import json
import sys
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Literal

from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from analysis_engine import analyze_earnings_async
from aws_fmp_db import get_cursor, get_transcript, get_historical_prices, get_market_timing

# Configuration
CONCURRENCY = 15
REPORT_INTERVAL_SECONDS = 300  # 5 minutes for larger batch
SAVE_INTERVAL = 100  # Save progress every 100 completed

# Type definitions
EarningsSession = Literal["BMO", "AMC", "UNKNOWN"]


def get_session_from_db(symbol: str, year: int, quarter: int) -> EarningsSession:
    """
    Get BMO/AMC session from AWS DB market_timing field.
    """
    timing = get_market_timing(symbol, year, quarter)
    if timing:
        timing_lower = timing.lower()
        if "before" in timing_lower:
            return "BMO"
        elif "after" in timing_lower:
            return "AMC"
    return "UNKNOWN"


def get_trading_day_price(symbol: str, target_date: str) -> Optional[float]:
    """Get close price for a specific date from AWS DB."""
    try:
        dt = datetime.strptime(target_date, "%Y-%m-%d")
        start = (dt - timedelta(days=5)).strftime("%Y-%m-%d")
        end = (dt + timedelta(days=5)).strftime("%Y-%m-%d")

        prices = get_historical_prices(symbol, start, end)
        if not prices:
            return None

        for p in prices:
            if p.get("date") == target_date:
                return p.get("close")

        return None
    except Exception:
        return None


def get_next_trading_day_price(symbol: str, after_date: str) -> tuple[Optional[str], Optional[float]]:
    """Get the next trading day's close price after the given date."""
    try:
        dt = datetime.strptime(after_date, "%Y-%m-%d")
        start = after_date
        end = (dt + timedelta(days=10)).strftime("%Y-%m-%d")

        prices = get_historical_prices(symbol, start, end)
        if not prices:
            return None, None

        for p in prices:
            if p.get("date") > after_date:
                return p.get("date"), p.get("close")

        return None, None
    except Exception:
        return None, None


def get_t_plus_n_trading_day_price(symbol: str, base_date: str, n: int) -> tuple[Optional[str], Optional[float]]:
    """Get the price N trading days after base_date."""
    try:
        dt = datetime.strptime(base_date, "%Y-%m-%d")
        end = (dt + timedelta(days=n + 20)).strftime("%Y-%m-%d")

        prices = get_historical_prices(symbol, base_date, end)
        if not prices:
            return None, None

        trading_days_after = [p for p in prices if p.get("date") > base_date]

        if len(trading_days_after) >= n:
            target = trading_days_after[n - 1]
            return target.get("date"), target.get("close")

        return None, None
    except Exception:
        return None, None


def calculate_t30_change(
    symbol: str,
    year: int,
    quarter: int,
    earnings_date: str
) -> Dict:
    """
    Calculate T+30 change based on BMO/AMC session.
    """
    result = {
        "session": "UNKNOWN",
        "base_date": None,
        "base_price": None,
        "t30_date": None,
        "t30_price": None,
        "t30_change": None,
    }

    result["session"] = get_session_from_db(symbol, year, quarter)

    if result["session"] == "BMO":
        base_date = earnings_date
        base_price = get_trading_day_price(symbol, earnings_date)

        if base_price is None:
            next_date, next_price = get_next_trading_day_price(symbol, earnings_date)
            if next_date:
                base_date = next_date
                base_price = next_price
    else:
        base_date, base_price = get_next_trading_day_price(symbol, earnings_date)

    if not base_date or base_price is None:
        return result

    result["base_date"] = base_date
    result["base_price"] = base_price

    t30_date, t30_price = get_t_plus_n_trading_day_price(symbol, base_date, 30)

    if not t30_date or t30_price is None:
        return result

    result["t30_date"] = t30_date
    result["t30_price"] = t30_price

    if base_price and base_price != 0:
        result["t30_change"] = round((t30_price - base_price) / base_price * 100, 2)

    return result


def fetch_all_sp500_transcripts(years: List[int] = [2023, 2024], quarters: Optional[List[int]] = None) -> List[Dict]:
    """
    Fetch all S&P 500 transcripts for specified years/quarters from AWS DB.
    Returns list of dicts with: symbol, year, quarter, t_day, market_timing, sector
    """
    with get_cursor() as cur:
        if cur is None:
            print("ERROR: Cannot connect to AWS DB", flush=True)
            return []

        year_filter = ','.join(str(y) for y in years)

        query = f"""
            SELECT
                et.symbol,
                et.year,
                et.quarter,
                et.t_day::text as t_day,
                et.market_timing,
                c.sector,
                c.name as company_name
            FROM earnings_transcripts et
            JOIN companies c ON et.symbol = c.symbol
            WHERE et.year IN ({year_filter})
        """

        if quarters:
            quarter_filter = ','.join(str(q) for q in quarters)
            query += f" AND et.quarter IN ({quarter_filter})"

        query += " ORDER BY et.year, et.quarter, et.symbol"

        cur.execute(query)
        rows = cur.fetchall()

        results = []
        for row in rows:
            results.append({
                "symbol": row["symbol"],
                "year": row["year"],
                "quarter": row["quarter"],
                "t_day": row["t_day"],
                "market_timing": row["market_timing"],
                "sector": row["sector"],
                "company_name": row["company_name"],
            })

        return results


def determine_hit(prediction: str, actual_change: float) -> str:
    """Determine if prediction was a HIT, MISS, or SKIP"""
    if not prediction or prediction == "N/A":
        return "ERROR"

    pred_upper = prediction.upper()
    if pred_upper == "NEUTRAL":
        return "SKIP"

    if actual_change is None:
        return "NO_DATA"

    actual_direction = "UP" if actual_change > 0 else "DOWN"

    if pred_upper == actual_direction:
        return "HIT"
    else:
        return "MISS"


async def analyze_single(test_case: Dict, idx: int, total: int, semaphore: asyncio.Semaphore, results: List, progress: Dict):
    """Analyze a single test case"""
    async with semaphore:
        symbol = test_case["symbol"]
        year = test_case["year"]
        quarter = test_case["quarter"]
        t_day = test_case["t_day"]

        result_row = {
            "idx": idx,
            "symbol": symbol,
            "company_name": test_case.get("company_name"),
            "sector": test_case.get("sector"),
            "year": year,
            "quarter": quarter,
            "t_day": t_day,
            "session": None,
            "base_date": None,
            "base_price": None,
            "t30_date": None,
            "t30_price": None,
            "t30_change": None,
            "predicted": None,
            "confidence": None,
            "hit_result": None,
            "status": "pending",
            "error": None,
            "elapsed_seconds": None,
        }

        try:
            print(f"[{idx}/{total}] {symbol} {year}Q{quarter} ({test_case.get('sector', 'N/A')[:20]})...", flush=True)

            # Calculate T+30 change
            t30_calc = calculate_t30_change(symbol, year, quarter, t_day)
            result_row.update({
                "session": t30_calc["session"],
                "base_date": t30_calc["base_date"],
                "base_price": t30_calc["base_price"],
                "t30_date": t30_calc["t30_date"],
                "t30_price": t30_calc["t30_price"],
                "t30_change": t30_calc["t30_change"],
            })

            start = time.time()
            analysis_result = await analyze_earnings_async(
                symbol=symbol,
                year=year,
                quarter=quarter,
                skip_cache=True,
            )
            elapsed = time.time() - start
            result_row["elapsed_seconds"] = round(elapsed, 1)

            agentic_result = analysis_result.get("agentic_result", {}) if analysis_result else {}
            if analysis_result and agentic_result.get("prediction"):
                result_row["predicted"] = agentic_result.get("prediction", "N/A")
                result_row["confidence"] = agentic_result.get("confidence")
                result_row["hit_result"] = determine_hit(result_row["predicted"], t30_calc["t30_change"])
                result_row["status"] = "success"

                progress["success"] += 1
                if result_row["hit_result"] == "HIT":
                    progress["hits"] += 1
                elif result_row["hit_result"] == "MISS":
                    progress["misses"] += 1
                elif result_row["hit_result"] == "SKIP":
                    progress["skips"] += 1
                elif result_row["hit_result"] == "NO_DATA":
                    progress["no_data"] += 1
            else:
                result_row["status"] = "error"
                result_row["error"] = "No prediction returned"
                progress["errors"] += 1

        except Exception as e:
            result_row["status"] = "error"
            result_row["error"] = str(e)[:200]
            result_row["elapsed_seconds"] = time.time() - start if 'start' in dir() else None
            progress["errors"] += 1

        progress["processed"] += 1
        results.append(result_row)


async def progress_reporter(progress: Dict, total: int, start_time: float, results: List, csv_path: Path):
    """Report progress and save periodically"""
    last_save_count = 0

    while progress["processed"] < total:
        await asyncio.sleep(REPORT_INTERVAL_SECONDS)

        elapsed_min = (time.time() - start_time) / 60
        processed = progress["processed"]
        success = progress["success"]
        errors = progress["errors"]
        hits = progress["hits"]
        misses = progress["misses"]
        skips = progress["skips"]
        no_data = progress["no_data"]

        hit_rate = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0

        # Estimate remaining time
        if processed > 0:
            avg_time = elapsed_min / processed
            remaining = (total - processed) * avg_time
        else:
            remaining = 0

        print(f"\n{'='*100}", flush=True)
        print(f"PROGRESS REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print(f"Elapsed: {elapsed_min:.1f} min | Processed: {processed}/{total} ({processed/total*100:.1f}%)", flush=True)
        print(f"Estimated remaining: {remaining:.0f} min", flush=True)
        print(f"Success: {success} | Errors: {errors}", flush=True)
        print(f"HIT: {hits} | MISS: {misses} | SKIP: {skips} | NO_DATA: {no_data}", flush=True)
        print(f"Hit Rate: {hit_rate:.1f}%", flush=True)
        print(f"{'='*100}\n", flush=True)

        # Save intermediate results
        if processed - last_save_count >= SAVE_INTERVAL:
            save_intermediate_results(results, csv_path)
            last_save_count = processed
            print(f"  [Saved intermediate results to {csv_path}]", flush=True)


def save_intermediate_results(results: List, csv_path: Path):
    """Save current results to CSV"""
    sorted_results = sorted(results, key=lambda x: (x["year"], x["quarter"], x["symbol"]))

    csv_columns = [
        'idx', 'symbol', 'company_name', 'sector', 'year', 'quarter', 't_day',
        'session', 'base_date', 'base_price', 't30_date', 't30_price', 't30_change',
        'predicted', 'confidence', 'hit_result', 'status', 'elapsed_seconds', 'error'
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(sorted_results)


async def main():
    parser = argparse.ArgumentParser(description='Run S&P 500 batch test')
    parser.add_argument('--start-from', type=str, help='Start from specific symbol')
    parser.add_argument('--year', type=int, help='Filter by year (2023 or 2024)')
    parser.add_argument('--quarter', type=int, help='Filter by quarter (1-4)')
    args = parser.parse_args()

    years = [args.year] if args.year else [2023, 2024]
    quarters = [args.quarter] if args.quarter else None

    print("="*100)
    print("BATCH TEST - S&P 500 2023-2024 Quarterly Earnings")
    print(f"Years: {years}")
    print(f"Quarters: {quarters if quarters else 'All'}")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Report Interval: {REPORT_INTERVAL_SECONDS}s")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*100)
    print()

    # Fetch all test cases
    print("Fetching test cases from AWS DB...", flush=True)
    test_cases = fetch_all_sp500_transcripts(years=years, quarters=quarters)

    if not test_cases:
        print("ERROR: No test cases found!", flush=True)
        return []

    # Filter by start symbol if specified
    if args.start_from:
        start_idx = 0
        for i, tc in enumerate(test_cases):
            if tc["symbol"] == args.start_from.upper():
                start_idx = i
                break
        test_cases = test_cases[start_idx:]
        print(f"  Starting from {args.start_from} (index {start_idx})", flush=True)

    total = len(test_cases)
    print(f"  Loaded {total} test cases", flush=True)

    # Show breakdown by year-quarter
    breakdown = {}
    for tc in test_cases:
        key = f"{tc['year']}Q{tc['quarter']}"
        breakdown[key] = breakdown.get(key, 0) + 1

    print("\n--- Test Cases by Quarter ---")
    for key in sorted(breakdown.keys()):
        print(f"  {key}: {breakdown[key]} stocks")

    # Show sector distribution
    sectors = {}
    for tc in test_cases:
        s = tc.get("sector", "Unknown")
        sectors[s] = sectors.get(s, 0) + 1

    print("\n--- Test Cases by Sector ---")
    for sector, count in sorted(sectors.items(), key=lambda x: -x[1])[:10]:
        print(f"  {sector}: {count}")

    print("\n" + "="*100)
    print("Starting analysis...")
    print()

    # Initialize
    semaphore = asyncio.Semaphore(CONCURRENCY)
    results = []
    progress = {
        "processed": 0,
        "success": 0,
        "errors": 0,
        "hits": 0,
        "misses": 0,
        "skips": 0,
        "no_data": 0,
    }

    start_time = time.time()

    # Prepare output paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    year_str = f"{years[0]}" if len(years) == 1 else f"{years[0]}_{years[-1]}"
    quarter_str = f"Q{quarters[0]}" if quarters and len(quarters) == 1 else "all"

    json_path = Path(__file__).resolve().parent / f"batch_sp500_{year_str}_{quarter_str}_{timestamp}.json"
    csv_path = Path(__file__).resolve().parent / f"batch_sp500_{year_str}_{quarter_str}_{timestamp}.csv"

    # Start progress reporter
    reporter_task = asyncio.create_task(progress_reporter(progress, total, start_time, results, csv_path))

    # Run all analyses
    tasks = [
        analyze_single(tc, i + 1, total, semaphore, results, progress)
        for i, tc in enumerate(test_cases)
    ]
    await asyncio.gather(*tasks)

    # Cancel reporter
    reporter_task.cancel()
    try:
        await reporter_task
    except asyncio.CancelledError:
        pass

    elapsed = time.time() - start_time

    # Sort results by year, quarter, symbol
    results.sort(key=lambda x: (x["year"], x["quarter"], x["symbol"]))

    # Save final results
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    save_intermediate_results(results, csv_path)

    # Final summary
    success = progress["success"]
    errors = progress["errors"]
    hits = progress["hits"]
    misses = progress["misses"]
    skips = progress["skips"]
    no_data = progress["no_data"]

    # Breakdown by year-quarter
    print("\n" + "="*100)
    print("FINAL SUMMARY")
    print("="*100)
    print(f"Total: {total} | Success: {success} | Errors: {errors}")
    print(f"Total Time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
    print(f"Average per ticker: {elapsed/total:.1f}s")
    print()
    print(f"--- Overall Hit Rate ---")
    print(f"HIT: {hits} | MISS: {misses} | SKIP: {skips} | NO_DATA: {no_data}")
    hit_rate = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0
    print(f"Hit Rate (HIT/(HIT+MISS)): {hit_rate:.1f}%")
    print()

    # Breakdown by quarter
    print("--- Hit Rate by Quarter ---")
    for yq in sorted(breakdown.keys()):
        yq_results = [r for r in results if f"{r['year']}Q{r['quarter']}" == yq]
        yq_hits = sum(1 for r in yq_results if r["hit_result"] == "HIT")
        yq_misses = sum(1 for r in yq_results if r["hit_result"] == "MISS")
        yq_rate = yq_hits / (yq_hits + yq_misses) * 100 if (yq_hits + yq_misses) > 0 else 0
        print(f"  {yq}: HIT={yq_hits}, MISS={yq_misses}, Rate={yq_rate:.1f}%")

    # Breakdown by sector (top 5)
    print("\n--- Hit Rate by Sector (Top 10) ---")
    sector_stats = {}
    for r in results:
        s = r.get("sector", "Unknown")
        if s not in sector_stats:
            sector_stats[s] = {"hits": 0, "misses": 0, "total": 0}
        sector_stats[s]["total"] += 1
        if r["hit_result"] == "HIT":
            sector_stats[s]["hits"] += 1
        elif r["hit_result"] == "MISS":
            sector_stats[s]["misses"] += 1

    for sector, stats in sorted(sector_stats.items(), key=lambda x: -x[1]["total"])[:10]:
        s_rate = stats["hits"] / (stats["hits"] + stats["misses"]) * 100 if (stats["hits"] + stats["misses"]) > 0 else 0
        print(f"  {sector}: HIT={stats['hits']}, MISS={stats['misses']}, Rate={s_rate:.1f}%")

    print()
    print(f"Results saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV: {csv_path}")
    print("="*100)

    return results


if __name__ == "__main__":
    asyncio.run(main())
