#!/usr/bin/env python3
"""
Batch analysis script that calls the deployed Zeabur API.
Usage: python3 scripts/run_batch_via_api.py [--api-url URL]
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import aiohttp

# Default API URL (Zeabur deployment)
DEFAULT_API_URL = "https://earnings-call-check.zeabur.app"

# Top 100 Gainers (from PDF)
TOP_GAINERS = [
    "ORCL", "IDXX", "NRG", "APP", "PLTR", "DAL", "DDOG", "WST", "JBHT", "PODD",
    "CHRW", "GNRC", "STX", "CRL", "TTD", "EBAY", "CHRW", "IQV", "EXPE", "ANET",
    "MGM", "EXPE", "DHI", "AXON", "LW", "DXCM", "DG", "NOW", "CAH", "AXON",
    "NKE", "CVS", "GM", "IDXX", "AKAM", "GEV", "HAS", "ABNB", "FSLR", "AXON",
    "HOOD", "TTWO", "ISRG", "EFX", "ULTA", "ORCL", "IBM", "HAS", "DOW", "EPAM",
    "ROK", "GRMN", "MCHP", "INTU", "CNC", "HUM", "LVS", "KVUE", "F", "TPR",
    "RCL", "DAL", "APP", "TEL", "ROK", "APP", "GLW", "REGN", "ULTA", "AES",
    "CAT", "CARR", "HAL", "STX", "PHM", "CHTR", "FFIV", "DECK", "JCI", "META",
    "IDXX", "LVS", "PM", "WST", "EXPD", "HSIC", "DIS", "XYL", "NRG", "MPWR",
    "WYNN", "BEN", "ALLE", "CEG", "INCY", "WDC", "KEYS", "LW", "PWR", "KR",
]

# Top 100 Losers (from PDF)
TOP_LOSERS = [
    "FISV", "TTD", "WST", "ALGN", "SNPS", "TTD", "IT", "SWKS", "BAX", "UNH",
    "FTNT", "AKAM", "VRTX", "DECK", "XYZ", "DECK", "LULU", "ARE", "SRE", "LULU",
    "FISV", "CHTR", "HII", "SMCI", "CMG", "BDX", "LKQ", "XYZ", "DASH", "DOW",
    "CI", "STZ", "COIN", "EME", "EL", "EBAY", "TPR", "SJM", "ON", "NTAP",
    "NCLH", "DECK", "VTRS", "ZBH", "DXCM", "COO", "BAX", "GDDY", "LULU", "LLY",
    "UPS", "AMAT", "ADBE", "FISV", "ZTS", "NRG", "TXN", "CMG", "BBY", "PYPL",
    "HRL", "COO", "IP", "EPAM", "IP", "NOC", "WDAY", "OTIS", "VST", "ELV",
    "SW", "PLTR", "HPE", "TDG", "ZBRA", "LLY", "ZBH", "LKQ", "CPRT", "FIS",
    "GRMN", "NOW", "ZBRA", "META", "SMCI", "IEX", "GDDY", "TMUS", "DVA", "CMCSA",
    "J", "LMT", "HOOD", "AKAM", "PAYC", "CARR", "LYV", "UPS", "VRSK", "FDS",
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

# All unique tickers (145 after deduplication)
ALL_TICKERS = get_unique_tickers()


async def apply_profile(session: aiohttp.ClientSession, api_url: str, profile_name: str) -> bool:
    """Apply a profile via API."""
    url = f"{api_url}/api/prompt_profiles/apply"
    try:
        async with session.post(url, json={"name": profile_name}) as resp:
            if resp.status == 200:
                result = await resp.json()
                print(f"Profile '{profile_name}' applied: {result}")
                return True
            else:
                text = await resp.text()
                print(f"Failed to apply profile: {resp.status} - {text}")
                return False
    except Exception as e:
        print(f"Error applying profile: {e}")
        return False


async def start_batch_job(session: aiohttp.ClientSession, api_url: str, tickers: list) -> str:
    """Start a batch analysis job and return job_id."""
    url = f"{api_url}/api/batch-analyze"
    try:
        async with session.post(url, json={"tickers": tickers, "latest_only": True}) as resp:
            if resp.status == 200:
                result = await resp.json()
                job_id = result.get("job_id")
                print(f"Batch job started: {job_id} ({len(tickers)} tickers)")
                return job_id
            else:
                text = await resp.text()
                print(f"Failed to start batch job: {resp.status} - {text}")
                return None
    except Exception as e:
        print(f"Error starting batch job: {e}")
        return None


async def poll_job_status(session: aiohttp.ClientSession, api_url: str, job_id: str) -> dict:
    """Poll batch job status."""
    url = f"{api_url}/api/batch-analyze/{job_id}"
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                return None
    except Exception as e:
        print(f"Error polling job status: {e}")
        return None


def print_progress(job: dict, last_completed: int) -> int:
    """Print progress update, return new completed count."""
    completed = job.get("completed", 0)
    total = job.get("total", 0)
    current = job.get("current", "")
    status = job.get("status", "unknown")

    if completed > last_completed:
        results = job.get("results", [])
        # Print new results
        for r in results[last_completed:completed]:
            ticker = r.get("symbol", "?")
            r_status = r.get("status", "?")
            payload = r.get("payload", {})
            if r_status == "ok" and payload:
                agent_result = payload.get("agent_result", {})
                pred = agent_result.get("prediction", "N/A")
                conf = agent_result.get("confidence", "N/A")
                print(f"  {ticker}: {pred} (conf={conf})")
            else:
                error = r.get("error", "unknown error")
                print(f"  {ticker}: {r_status} - {error[:50]}")

    print(f"\n[{completed}/{total}] Status: {status}, Current: {current}")
    return completed


def print_summary(job: dict):
    """Print final summary."""
    print("\n" + "=" * 70)
    print("BATCH ANALYSIS SUMMARY")
    print("=" * 70)

    results = job.get("results", [])
    total = job.get("total", 0)

    success = [r for r in results if r.get("status") == "ok" and r.get("payload")]
    errors = [r for r in results if r.get("status") != "ok" or not r.get("payload")]

    print(f"\nTotal: {total}")
    print(f"Success: {len(success)}")
    print(f"Errors: {len(errors)}")

    if success:
        up_pred = []
        down_pred = []
        for r in success:
            payload = r.get("payload", {})
            agent_result = payload.get("agent_result", {})
            pred = agent_result.get("prediction", "")
            conf = agent_result.get("confidence", 0.5)
            symbol = r.get("symbol", "?")

            if pred == "UP":
                up_pred.append({"ticker": symbol, "confidence": conf})
            elif pred == "DOWN":
                down_pred.append({"ticker": symbol, "confidence": conf})

        print(f"\nPredictions:")
        print(f"  UP: {len(up_pred)}")
        print(f"  DOWN: {len(down_pred)}")

        # High confidence predictions
        high_conf_up = [r for r in up_pred if r.get("confidence", 0) >= 0.7]
        high_conf_down = [r for r in down_pred if r.get("confidence", 1) <= 0.3]

        print(f"\n=== HIGH CONFIDENCE SIGNALS ===")
        print(f"\nLONG (UP with conf >= 0.7): {len(high_conf_up)}")
        if high_conf_up:
            for r in sorted(high_conf_up, key=lambda x: x.get("confidence", 0), reverse=True):
                print(f"    {r['ticker']}: conf={r.get('confidence')}")

        print(f"\nSHORT (DOWN with conf <= 0.3): {len(high_conf_down)}")
        if high_conf_down:
            for r in sorted(high_conf_down, key=lambda x: x.get("confidence", 1)):
                print(f"    {r['ticker']}: conf={r.get('confidence')}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for r in errors[:20]:
            print(f"  {r.get('symbol', '?')}: {r.get('status')} - {r.get('error', 'N/A')[:40]}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")


async def run_batch_with_monitoring(api_url: str, tickers: list, profile_name: str = "garen1204"):
    """Run batch analysis with progress monitoring."""
    print("=" * 70)
    print("BATCH ANALYSIS VIA ZEABUR API")
    print(f"API: {api_url}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    connector = aiohttp.TCPConnector(limit=5)
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Step 1: Apply profile
        print(f"\nApplying profile '{profile_name}'...")
        if not await apply_profile(session, api_url, profile_name):
            print("Warning: Could not apply profile, continuing with current settings...")

        # Step 2: Start batch job
        print(f"\nStarting batch analysis for {len(tickers)} tickers...")
        job_id = await start_batch_job(session, api_url, tickers)
        if not job_id:
            print("Failed to start batch job!")
            return None

        # Step 3: Poll for progress
        print("\nMonitoring progress...\n")
        last_completed = 0
        start_time = time.time()

        while True:
            job = await poll_job_status(session, api_url, job_id)
            if not job:
                print("Lost connection to job, retrying...")
                await asyncio.sleep(5)
                continue

            status = job.get("status", "unknown")
            last_completed = print_progress(job, last_completed)

            if status in ("done", "error"):
                break

            await asyncio.sleep(10)  # Poll every 10 seconds

        # Step 4: Final summary
        elapsed = time.time() - start_time
        print(f"\nTotal elapsed time: {elapsed:.1f}s")
        print_summary(job)

        # Save results
        output_file = Path(__file__).parent / f"api_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(job, f, indent=2)
        print(f"\nResults saved to: {output_file}")

        return job


async def main():
    parser = argparse.ArgumentParser(description="Run batch analysis via API")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="API base URL")
    parser.add_argument("--profile", default="garen1204", help="Profile name to apply")
    parser.add_argument("--batch-size", type=int, default=50, help="Tickers per batch")
    args = parser.parse_args()

    tickers = ALL_TICKERS
    print(f"Total tickers: {len(tickers)}")

    # Run in batches for better reliability
    batch_size = args.batch_size
    all_results = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(tickers) + batch_size - 1) // batch_size

        print(f"\n{'='*70}")
        print(f"BATCH {batch_num}/{total_batches}")
        print(f"{'='*70}")

        job = await run_batch_with_monitoring(args.api_url, batch, args.profile)
        if job:
            all_results.extend(job.get("results", []))

        # Wait between batches to avoid overwhelming the API
        if i + batch_size < len(tickers):
            print("\nWaiting 30 seconds before next batch...")
            await asyncio.sleep(30)

    # Final combined summary
    print("\n" + "=" * 70)
    print("COMBINED RESULTS FROM ALL BATCHES")
    print("=" * 70)

    success = [r for r in all_results if r.get("status") == "ok"]
    print(f"Total analyzed: {len(all_results)}")
    print(f"Successful: {len(success)}")


if __name__ == "__main__":
    asyncio.run(main())
