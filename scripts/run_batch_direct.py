#!/usr/bin/env python3
"""
Direct batch analysis script using garen1204 profile.
Runs with CONCURRENCY=5 for parallel execution.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from analysis_engine import analyze_earnings_async
from storage import set_prompt, get_prompt_profile
from fmp_client import get_transcript_dates

# Concurrency setting
CONCURRENCY = 5

# Top 100 Gainers (from PDF - 2025-01-01 to 2025-12-02)
TOP_GAINERS = [
    "ORCL", "IDXX", "NRG", "APP", "PLTR", "DAL", "DDOG", "WST", "JBHT", "PODD",
    "CHRW", "GNRC", "STX", "CRL", "TTD", "EBAY", "IQV", "EXPE", "ANET", "MGM",
    "DHI", "AXON", "LW", "DXCM", "DG", "NOW", "CAH", "NKE", "CVS", "GM",
    "AKAM", "GEV", "HAS", "ABNB", "FSLR", "HOOD", "TTWO", "ISRG", "EFX", "ULTA",
    "IBM", "DOW", "EPAM", "ROK", "GRMN", "MCHP", "INTU", "CNC", "HUM", "LVS",
    "KVUE", "F", "TPR", "RCL", "TEL", "GLW", "REGN", "AES", "CAT", "CARR",
    "HAL", "PHM", "CHTR", "FFIV", "DECK", "JCI", "META", "PM", "EXPD", "HSIC",
    "DIS", "XYL", "MPWR", "WYNN", "BEN", "ALLE", "CEG", "INCY", "WDC", "KEYS",
    "PWR", "KR",
]

# Top 100 Losers (from PDF - 2025-01-01 to 2025-12-02)
TOP_LOSERS = [
    "FISV", "ALGN", "SNPS", "IT", "SWKS", "BAX", "UNH", "FTNT", "VRTX", "XYZ",
    "LULU", "ARE", "SRE", "HII", "SMCI", "CMG", "BDX", "LKQ", "DASH", "CI",
    "STZ", "COIN", "EME", "EL", "SJM", "ON", "NTAP", "NCLH", "VTRS", "ZBH",
    "COO", "GDDY", "LLY", "UPS", "AMAT", "ADBE", "ZTS", "TXN", "BBY", "PYPL",
    "HRL", "IP", "NOC", "WDAY", "OTIS", "VST", "ELV", "SW", "HPE", "TDG",
    "ZBRA", "CPRT", "FIS", "IEX", "TMUS", "DVA", "CMCSA", "J", "LMT", "PAYC",
    "LYV", "VRSK", "FDS",
]

def get_unique_tickers():
    """Remove duplicates while preserving order."""
    seen = set()
    unique = []
    for t in TOP_GAINERS + TOP_LOSERS:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def apply_profile(profile_name: str) -> bool:
    """Apply a saved profile to current settings."""
    profile = get_prompt_profile(profile_name)
    if not profile:
        print(f"ERROR: Profile '{profile_name}' not found!")
        return False

    prompts = profile.get("prompts", {})
    for key, content in prompts.items():
        set_prompt(key, content)
    print(f"Applied profile '{profile_name}' with {len(prompts)} prompts")
    return True


def get_latest_quarter(ticker: str):
    """Get the most recent available quarter for a ticker."""
    try:
        dates = get_transcript_dates(ticker)
        valid = []
        for d in dates:
            y = d.get("year") or d.get("calendar_year")
            q = d.get("quarter") or d.get("calendar_quarter")
            if y is None or q is None:
                continue
            try:
                valid.append((int(y), int(q)))
            except Exception:
                continue
        if valid:
            valid.sort(reverse=True)
            return valid[0]
    except Exception as e:
        print(f"  Warning: Could not get quarters for {ticker}: {e}")
    return None, None


# Global progress tracking
progress_lock = asyncio.Lock()
completed_count = 0
results_list = []


async def analyze_single(ticker: str, idx: int, total: int, semaphore: asyncio.Semaphore) -> dict:
    """Analyze a single ticker with semaphore for concurrency control."""
    global completed_count, results_list

    async with semaphore:
        start_time = time.time()
        print(f"\n[{idx}/{total}] Analyzing {ticker}...")

        try:
            result = get_latest_quarter(ticker)
            if not result or result == (None, None):
                return {
                    "ticker": ticker,
                    "status": "NO_QUARTERS",
                    "error": "No available quarters found",
                    "elapsed": 0,
                }
            year, quarter = result
            if not year or not quarter:
                return {
                    "ticker": ticker,
                    "status": "NO_QUARTERS",
                    "error": "No available quarters found",
                    "elapsed": 0,
                }

            print(f"  {ticker}: Quarter {year}-Q{quarter}")
            result = await analyze_earnings_async(ticker, year, quarter, skip_cache=True)

            elapsed = time.time() - start_time

            if result and "error" not in result:
                # prediction and confidence are inside agentic_result
                agentic_result = result.get("agentic_result", {})
                prediction = agentic_result.get("prediction", "N/A") if isinstance(agentic_result, dict) else "N/A"
                confidence = agentic_result.get("confidence", "N/A") if isinstance(agentic_result, dict) else "N/A"

                print(f"  {ticker}: {prediction} (conf={confidence}) [{elapsed:.1f}s]")

                return {
                    "ticker": ticker,
                    "status": "success",
                    "year": year,
                    "quarter": quarter,
                    "predicted": prediction,
                    "confidence": confidence,
                    "elapsed": elapsed,
                }
            else:
                error_msg = result.get("error", "Unknown error") if result else "No result"
                print(f"  {ticker}: ERROR - {error_msg[:50]}")
                return {
                    "ticker": ticker,
                    "status": "ERROR",
                    "error": error_msg,
                    "elapsed": elapsed,
                }

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  {ticker}: EXCEPTION - {str(e)[:50]}")
            return {
                "ticker": ticker,
                "status": "EXCEPTION",
                "error": str(e),
                "elapsed": elapsed,
            }


def print_summary(results: list):
    """Print a summary of all results."""
    print("\n" + "=" * 70)
    print("BATCH ANALYSIS SUMMARY")
    print("=" * 70)

    success = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] != "success"]

    print(f"\nTotal: {len(results)}")
    print(f"Success: {len(success)}")
    print(f"Errors: {len(errors)}")

    if success:
        up_pred = [r for r in success if r.get("predicted") == "UP"]
        down_pred = [r for r in success if r.get("predicted") == "DOWN"]

        print(f"\nPredictions:")
        print(f"  UP: {len(up_pred)}")
        print(f"  DOWN: {len(down_pred)}")

        # High confidence predictions (>= 0.7 for UP, <= 0.3 for DOWN)
        high_conf_up = [r for r in up_pred if r.get("confidence", 0) >= 0.7]
        high_conf_down = [r for r in down_pred if r.get("confidence", 0) <= 0.3]

        print(f"\nHigh Confidence Signals:")
        print(f"  LONG (UP >= 0.7): {len(high_conf_up)}")
        if high_conf_up:
            for r in sorted(high_conf_up, key=lambda x: x.get("confidence", 0), reverse=True)[:10]:
                print(f"    {r['ticker']}: conf={r.get('confidence')}")

        print(f"  SHORT (DOWN <= 0.3): {len(high_conf_down)}")
        if high_conf_down:
            for r in sorted(high_conf_down, key=lambda x: x.get("confidence", 0))[:10]:
                print(f"    {r['ticker']}: conf={r.get('confidence')}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for r in errors[:10]:
            print(f"  {r['ticker']}: {r.get('status')} - {r.get('error', 'N/A')[:50]}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    # Total time
    total_time = sum(r.get("elapsed", 0) for r in results)
    avg_time = total_time / len(results) if results else 0
    print(f"\nTotal elapsed time: {total_time:.1f}s")
    print(f"Average per ticker: {avg_time:.1f}s")
    print(f"Effective time (with concurrency {CONCURRENCY}): ~{total_time/CONCURRENCY:.1f}s")


async def main():
    print("=" * 70)
    print("BATCH ANALYSIS WITH GAREN1204 PROFILE")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Apply profile
    if not apply_profile("garen1204"):
        sys.exit(1)

    tickers = get_unique_tickers()
    total = len(tickers)
    print(f"\nTotal unique tickers: {total}")
    print(f"  (from {len(TOP_GAINERS)} gainers + {len(TOP_LOSERS)} losers)")

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(CONCURRENCY)

    # Create all tasks
    tasks = [
        analyze_single(ticker, idx, total, semaphore)
        for idx, ticker in enumerate(tickers, 1)
    ]

    # Run all tasks concurrently (semaphore limits actual concurrency)
    print(f"\nRunning {total} analyses with concurrency={CONCURRENCY}...\n")
    results = await asyncio.gather(*tasks)

    # Final summary
    print_summary(results)

    # Save results to file
    output_file = Path(__file__).parent / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    asyncio.run(main())
