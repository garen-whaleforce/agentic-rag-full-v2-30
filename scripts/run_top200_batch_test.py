#!/usr/bin/env python3
"""
Batch test for top 200 earnings moves from CSV file.
Uses the provided CSV with pre-calculated T+30 data.
"""

import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from analysis_engine import analyze_earnings_async

# Configuration
CONCURRENCY = 15
REPORT_INTERVAL_SECONDS = 120  # 2 minutes for larger batch
CSV_FILE = Path(__file__).resolve().parent / "top200_earnings_moves.csv"


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


def load_test_cases() -> List[Dict]:
    """Load test cases from CSV file"""
    test_cases = []
    
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rank = int(row['rank'])
            symbol = row['symbol']
            earnings_date = row['earnings_date']
            pct_change = parse_pct(row['pct_change'])
            session = "BMO" if row['time_of_day'] == 'bmo' else "AMC"
            
            # Determine fiscal year/quarter
            fiscal_year, fiscal_quarter = determine_fiscal_quarter(earnings_date)
            
            # Determine category (gainer/loser based on rank or pct_change)
            category = "GAINER" if pct_change > 0 else "LOSER"
            
            test_cases.append({
                "rank": rank,
                "symbol": symbol,
                "year": fiscal_year,
                "quarter": fiscal_quarter,
                "earnings_date": earnings_date,
                "session": session,
                "category": category,
                "csv_t30_change": pct_change,
                "from_date": row['from_date'],
                "to_date": row['to_date'],
                "from_close": float(row['from_close']) if row['from_close'] else None,
                "to_close": float(row['to_close']) if row['to_close'] else None,
            })
    
    return test_cases


async def analyze_single(test_case: Dict, semaphore: asyncio.Semaphore, results: List, progress: Dict):
    """Analyze a single test case"""
    async with semaphore:
        rank = test_case["rank"]
        symbol = test_case["symbol"]
        year = test_case["year"]
        quarter = test_case["quarter"]
        
        result_row = {
            **test_case,
            "predicted": None,
            "confidence": None,
            "hit_result": None,
            "status": "pending",
            "error": None,
            "elapsed_seconds": None,
        }
        
        try:
            print(f"[{rank}/200] {symbol} {year}-Q{quarter} ({test_case['category']})...", flush=True)
            
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
            agentic_result = analysis_result.get("agentic_result", {}) if analysis_result else {}
            if analysis_result and agentic_result.get("prediction"):
                result_row["predicted"] = agentic_result.get("prediction", "N/A")
                result_row["confidence"] = agentic_result.get("confidence")
                result_row["hit_result"] = determine_hit(result_row["predicted"], test_case["csv_t30_change"])
                result_row["status"] = "success"
                
                progress["success"] += 1
                if result_row["hit_result"] == "HIT":
                    progress["hits"] += 1
                elif result_row["hit_result"] == "MISS":
                    progress["misses"] += 1
                elif result_row["hit_result"] == "SKIP":
                    progress["skips"] += 1
            else:
                result_row["status"] = "error"
                result_row["error"] = "No prediction returned"
                progress["errors"] += 1
                
        except Exception as e:
            result_row["status"] = "error"
            result_row["error"] = str(e)
            result_row["elapsed_seconds"] = time.time() - start if 'start' in dir() else None
            progress["errors"] += 1
        
        progress["processed"] += 1
        results.append(result_row)


async def progress_reporter(progress: Dict, total: int, start_time: float):
    """Report progress periodically"""
    while progress["processed"] < total:
        await asyncio.sleep(REPORT_INTERVAL_SECONDS)
        
        elapsed_min = (time.time() - start_time) / 60
        processed = progress["processed"]
        success = progress["success"]
        errors = progress["errors"]
        hits = progress["hits"]
        misses = progress["misses"]
        skips = progress["skips"]
        
        hit_rate = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0
        
        print(f"\n{'='*80}", flush=True)
        print(f"PROGRESS REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print(f"Elapsed: {elapsed_min:.1f} min | Processed: {processed}/{total}", flush=True)
        print(f"Success: {success} | Errors: {errors}", flush=True)
        print(f"HIT: {hits} | MISS: {misses} | SKIP: {skips} | Hit Rate: {hit_rate:.1f}%", flush=True)
        print(f"{'='*80}\n", flush=True)


async def main():
    print("="*100)
    print("BATCH TEST - Top 200 Earnings Moves from CSV")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Report Interval: {REPORT_INTERVAL_SECONDS}s")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*100)
    print()
    
    # Load test cases
    print("Loading test cases from CSV...", flush=True)
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
    }
    
    start_time = time.time()
    
    # Start progress reporter
    reporter_task = asyncio.create_task(progress_reporter(progress, total, start_time))
    
    # Run all analyses
    tasks = [analyze_single(tc, semaphore, results, progress) for tc in test_cases]
    await asyncio.gather(*tasks)
    
    # Cancel reporter
    reporter_task.cancel()
    try:
        await reporter_task
    except asyncio.CancelledError:
        pass
    
    elapsed = time.time() - start_time
    
    # Sort results by rank
    results.sort(key=lambda x: x["rank"])
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = Path(__file__).resolve().parent / f"batch_200_results_{timestamp}.json"
    csv_path = Path(__file__).resolve().parent / f"batch_200_results_{timestamp}.csv"
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save CSV
    csv_columns = [
        'rank', 'symbol', 'year', 'quarter', 'earnings_date', 'category', 'session',
        'predicted', 'confidence', 'csv_t30_change', 'hit_result', 'status',
        'elapsed_seconds', 'from_date', 'to_date', 'from_close', 'to_close', 'error'
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    # Final summary
    success = progress["success"]
    errors = progress["errors"]
    hits = progress["hits"]
    misses = progress["misses"]
    skips = progress["skips"]
    
    # Category breakdown
    gainer_results = [r for r in results if r["category"] == "GAINER"]
    loser_results = [r for r in results if r["category"] == "LOSER"]
    
    gainer_hits = sum(1 for r in gainer_results if r["hit_result"] == "HIT")
    gainer_misses = sum(1 for r in gainer_results if r["hit_result"] == "MISS")
    gainer_skips = sum(1 for r in gainer_results if r["hit_result"] == "SKIP")
    
    loser_hits = sum(1 for r in loser_results if r["hit_result"] == "HIT")
    loser_misses = sum(1 for r in loser_results if r["hit_result"] == "MISS")
    loser_skips = sum(1 for r in loser_results if r["hit_result"] == "SKIP")
    
    print("\n" + "="*100)
    print("FINAL SUMMARY")
    print("="*100)
    print(f"Total: {total} | Success: {success} | Errors: {errors}")
    print(f"Total Time: {elapsed/60:.1f} minutes")
    print(f"Average per ticker: {elapsed/total:.1f}s")
    print()
    print(f"--- Overall Hit Rate ---")
    print(f"HIT: {hits} | MISS: {misses} | SKIP: {skips}")
    hit_rate = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0
    print(f"Hit Rate (HIT/(HIT+MISS)): {hit_rate:.1f}%")
    print()
    print(f"--- GAINERS ({len(gainer_results)} total) ---")
    print(f"HIT: {gainer_hits} | MISS: {gainer_misses} | SKIP: {gainer_skips}")
    gainer_rate = gainer_hits / (gainer_hits + gainer_misses) * 100 if (gainer_hits + gainer_misses) > 0 else 0
    print(f"Hit Rate: {gainer_rate:.1f}%")
    print()
    print(f"--- LOSERS ({len(loser_results)} total) ---")
    print(f"HIT: {loser_hits} | MISS: {loser_misses} | SKIP: {loser_skips}")
    loser_rate = loser_hits / (loser_hits + loser_misses) * 100 if (loser_hits + loser_misses) > 0 else 0
    print(f"Hit Rate: {loser_rate:.1f}%")
    print()
    print(f"Results saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV: {csv_path}")
    print("="*100)
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
