#!/usr/bin/env python3
"""
Retry failed tickers from batch 200 test.
Uses closest available quarter if specific quarter not found.
"""

import sys
import os
os.environ['PYTHONUNBUFFERED'] = '1'

import asyncio
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from analysis_engine import analyze_earnings_async
from storage import set_prompt, get_prompt_profile
from fmp_client import get_transcript_dates

CONCURRENCY = 3

# Failed tickers with their expected data
FAILED_TICKERS = [
    {"rank": 58, "ticker": "KVUE", "date": "2025-11-03", "change": 12.32, "category": "GAINER", "expected": "UP"},
    {"rank": 73, "ticker": "HAL", "date": "2025-10-21", "change": 11.58, "category": "GAINER", "expected": "UP"},
    {"rank": 85, "ticker": "EXPD", "date": "2025-11-04", "change": 10.84, "category": "GAINER", "expected": "UP"},
    {"rank": 118, "ticker": "ARE", "date": "2025-10-27", "change": -19.17, "category": "LOSER", "expected": "DOWN"},
    {"rank": 139, "ticker": "ON", "date": "2025-08-04", "change": -15.58, "category": "LOSER", "expected": "DOWN"},
    {"rank": 142, "ticker": "DECK", "date": "2025-10-23", "change": -15.21, "category": "LOSER", "expected": "DOWN"},
    {"rank": 143, "ticker": "VTRS", "date": "2025-02-27", "change": -15.21, "category": "LOSER", "expected": "DOWN"},
    {"rank": 144, "ticker": "ZBH", "date": "2025-11-05", "change": -15.15, "category": "LOSER", "expected": "DOWN"},
    {"rank": 145, "ticker": "DXCM", "date": "2025-10-30", "change": -14.63, "category": "LOSER", "expected": "DOWN"},
    {"rank": 146, "ticker": "COO", "date": "2025-05-29", "change": -14.61, "category": "LOSER", "expected": "DOWN"},
    {"rank": 147, "ticker": "BAX", "date": "2025-10-30", "change": -14.54, "category": "LOSER", "expected": "DOWN"},
    {"rank": 148, "ticker": "GDDY", "date": "2025-02-13", "change": -14.28, "category": "LOSER", "expected": "DOWN"},
    {"rank": 149, "ticker": "LULU", "date": "2025-03-27", "change": -14.19, "category": "LOSER", "expected": "DOWN"},
    {"rank": 150, "ticker": "LLY", "date": "2025-08-07", "change": -14.14, "category": "LOSER", "expected": "DOWN"},
    {"rank": 151, "ticker": "UPS", "date": "2025-01-30", "change": -14.11, "category": "LOSER", "expected": "DOWN"},
    {"rank": 152, "ticker": "AMAT", "date": "2025-08-14", "change": -14.07, "category": "LOSER", "expected": "DOWN"},
    {"rank": 153, "ticker": "ADBE", "date": "2025-03-12", "change": -13.85, "category": "LOSER", "expected": "DOWN"},
    {"rank": 154, "ticker": "FISV", "date": "2025-07-23", "change": -13.84, "category": "LOSER", "expected": "DOWN"},
    {"rank": 155, "ticker": "ZTS", "date": "2025-11-04", "change": -13.78, "category": "LOSER", "expected": "DOWN"},
    {"rank": 156, "ticker": "NRG", "date": "2025-08-06", "change": -13.61, "category": "LOSER", "expected": "DOWN"},
    {"rank": 157, "ticker": "TXN", "date": "2025-07-22", "change": -13.34, "category": "LOSER", "expected": "DOWN"},
    {"rank": 158, "ticker": "CMG", "date": "2025-07-23", "change": -13.34, "category": "LOSER", "expected": "DOWN"},
    {"rank": 159, "ticker": "BBY", "date": "2025-03-04", "change": -13.3, "category": "LOSER", "expected": "DOWN"},
]


def apply_profile(profile_name: str) -> bool:
    profile = get_prompt_profile(profile_name)
    if not profile:
        print(f"ERROR: Profile '{profile_name}' not found!")
        return False
    prompts = profile.get("prompts", {})
    for key, content in prompts.items():
        set_prompt(key, content)
    print(f"Applied profile '{profile_name}' with {len(prompts)} prompts")
    return True


def get_best_quarter(ticker: str, target_date: str) -> Optional[tuple]:
    """Find the best available quarter closest to target date."""
    try:
        dates = get_transcript_dates(ticker)
        if not dates:
            return None
        
        # Parse target date
        from datetime import datetime as dt
        target = dt.strptime(target_date, "%Y-%m-%d")
        target_year = target.year
        target_month = target.month
        
        # Estimate target quarter
        if target_month <= 3:
            target_q = 4
            target_fy = target_year - 1
        elif target_month <= 6:
            target_q = 1
            target_fy = target_year
        elif target_month <= 9:
            target_q = 2
            target_fy = target_year
        else:
            target_q = 3
            target_fy = target_year
        
        # Find closest available quarter
        valid = []
        for d in dates:
            y = d.get("year") or d.get("calendar_year")
            q = d.get("quarter") or d.get("calendar_quarter")
            if y and q:
                valid.append((int(y), int(q)))
        
        if not valid:
            return None
        
        # Sort by closeness to target
        def distance(yq):
            y, q = yq
            return abs((y * 4 + q) - (target_fy * 4 + target_q))
        
        valid.sort(key=distance)
        return valid[0]  # Return closest quarter
        
    except Exception as e:
        print(f"  Error getting quarters for {ticker}: {e}")
        return None


def determine_result(predicted: str, actual: float) -> str:
    if predicted in ("N/A", "UNKNOWN", None, ""):
        return "N/A"
    pred_upper = str(predicted).upper()
    if pred_upper == "UP":
        return "CORRECT" if actual > 0 else "WRONG"
    elif pred_upper == "DOWN":
        return "CORRECT" if actual < 0 else "WRONG"
    elif pred_upper == "NEUTRAL":
        return "SKIP"
    return "N/A"


async def analyze_single(item: dict, idx: int, total: int, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        ticker = item["ticker"]
        rank = item["rank"]
        start_time = time.time()
        
        print(f"\n[{idx}/{total}] #{rank} {ticker} ({item['category']})...")
        
        try:
            quarter_info = get_best_quarter(ticker, item["date"])
            if not quarter_info:
                print(f"  {ticker}: NO_TRANSCRIPT - No quarters available")
                return {
                    "rank": rank,
                    "ticker": ticker,
                    "year": "N/A",
                    "quarter": "N/A",
                    "date": item["date"],
                    "category": item["category"],
                    "expected": item["expected"],
                    "predicted": "N/A",
                    "confidence": "N/A",
                    "actual": f"{item['change']:+.2f}%",
                    "result": "NO_DATA",
                    "elapsed": int(time.time() - start_time),
                }
            
            year, quarter = quarter_info
            print(f"  {ticker}: Using FY{year} Q{quarter}")
            
            result = await analyze_earnings_async(ticker, year, quarter, skip_cache=True)
            elapsed = int(time.time() - start_time)
            
            if result and "error" not in result:
                agentic_result = result.get("agentic_result", {})
                predicted = agentic_result.get("prediction", "N/A") if isinstance(agentic_result, dict) else "N/A"
                confidence = agentic_result.get("confidence", "N/A") if isinstance(agentic_result, dict) else "N/A"
                
                actual_change = item["change"]
                result_status = determine_result(predicted, actual_change)
                
                print(f"  {ticker}: {predicted} conf={confidence} actual={actual_change:+.2f}% -> {result_status} [{elapsed}s]")
                
                return {
                    "rank": rank,
                    "ticker": ticker,
                    "year": year,
                    "quarter": quarter,
                    "date": item["date"],
                    "category": item["category"],
                    "expected": item["expected"],
                    "predicted": predicted,
                    "confidence": confidence,
                    "actual": f"{actual_change:+.2f}%",
                    "result": result_status,
                    "elapsed": elapsed,
                }
            else:
                error_msg = result.get("error", "Unknown") if result else "No result"
                print(f"  {ticker}: ERROR - {error_msg[:50]}")
                return {
                    "rank": rank,
                    "ticker": ticker,
                    "year": year,
                    "quarter": quarter,
                    "date": item["date"],
                    "category": item["category"],
                    "expected": item["expected"],
                    "predicted": "ERROR",
                    "confidence": "N/A",
                    "actual": f"{item['change']:+.2f}%",
                    "result": "ERROR",
                    "elapsed": elapsed,
                }
                
        except Exception as e:
            elapsed = int(time.time() - start_time)
            print(f"  {ticker}: EXCEPTION - {str(e)[:60]}")
            return {
                "rank": rank,
                "ticker": ticker,
                "year": "N/A",
                "quarter": "N/A",
                "date": item["date"],
                "category": item["category"],
                "expected": item["expected"],
                "predicted": "EXCEPTION",
                "confidence": "N/A",
                "actual": f"{item['change']:+.2f}%",
                "result": "ERROR",
                "elapsed": elapsed,
            }


async def main():
    print("=" * 70)
    print("RETRY BATCH - 23 FAILED TICKERS")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)
    
    if not apply_profile("garen1204"):
        sys.exit(1)
    
    total = len(FAILED_TICKERS)
    print(f"\nRetrying {total} failed tickers...")
    
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [
        analyze_single(item, idx, total, semaphore)
        for idx, item in enumerate(FAILED_TICKERS, 1)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Print summary
    print("\n" + "=" * 70)
    print("RETRY RESULTS")
    print("=" * 70)
    
    success = [r for r in results if r["result"] not in ("ERROR", "NO_DATA")]
    errors = [r for r in results if r["result"] in ("ERROR", "NO_DATA")]
    
    print(f"Success: {len(success)}")
    print(f"Errors: {len(errors)}")
    
    if success:
        correct = sum(1 for r in success if r["result"] == "CORRECT")
        wrong = sum(1 for r in success if r["result"] == "WRONG")
        skip = sum(1 for r in success if r["result"] == "SKIP")
        print(f"  CORRECT: {correct}, WRONG: {wrong}, SKIP: {skip}")
    
    if errors:
        print(f"\nStill failed ({len(errors)}):")
        for r in errors:
            print(f"  #{r['rank']} {r['ticker']}: {r['result']}")
    
    # Save retry results
    csv_file = Path(__file__).parent / f"batch_retry_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['rank','ticker','year','quarter','date','category','expected','predicted','confidence','actual','result','elapsed'])
        writer.writeheader()
        for r in sorted(results, key=lambda x: x['rank']):
            writer.writerow(r)
    
    print(f"\nRetry results saved to: {csv_file}")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
