#!/usr/bin/env python3
"""
Batch analysis script for top gainers and losers using garen1204 profile.
Extracted from PDF data.
"""

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

# Top 50 Losers (from PDF)
TOP_LOSERS = [
    "FISV", "TTD", "WST", "ALGN", "SNPS", "TTD", "IT", "SWKS", "BAX", "UNH",
    "FTNT", "AKAM", "VRTX", "DECK", "XYZ", "DECK", "LULU", "ARE", "SRE", "LULU",
    "FISV", "CHTR", "HII", "SMCI", "CMG", "BDX", "LKQ", "XYZ", "DASH", "DOW",
    "CI", "STZ", "COIN", "EME", "EL", "EBAY", "TPR", "SJM", "ON", "NTAP",
    "NCLH", "DECK", "VTRS", "ZBH", "DXCM", "COO", "BAX", "GDDY", "LULU", "LLY",
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


def main():
    from prompt_service import get_prompt_profile, save_prompt_override
    from storage import get_all_prompts, set_prompt

    # Step 1: Apply garen1204 profile
    print("=" * 60)
    print("Step 1: Applying garen1204 profile...")
    print("=" * 60)

    profile = get_prompt_profile("garen1204")
    if not profile:
        print("ERROR: garen1204 profile not found!")
        sys.exit(1)

    prompts = profile.get("prompts", {})
    for key, content in prompts.items():
        set_prompt(key, content)
        print(f"  Applied: {key}")

    print(f"\nProfile 'garen1204' applied with {len(prompts)} prompts.")

    # Step 2: Prepare ticker list
    print("\n" + "=" * 60)
    print("Step 2: Preparing ticker list...")
    print("=" * 60)

    unique_tickers = get_unique_tickers()
    print(f"Total unique tickers: {len(unique_tickers)}")
    print(f"From gainers: {len(set(TOP_GAINERS))} unique")
    print(f"From losers: {len(set(TOP_LOSERS))} unique")

    # Step 3: Output for batch API call
    print("\n" + "=" * 60)
    print("Step 3: Batch Analysis Ready")
    print("=" * 60)

    print("\nTo run the batch analysis, use one of these methods:\n")

    print("Option A - Using curl:")
    print("-" * 40)
    ticker_json = json.dumps(unique_tickers[:50])  # First batch
    print(f'''curl -X POST "http://localhost:8000/api/batch-analyze" \\
  -H "Content-Type: application/json" \\
  -d '{{"tickers": {ticker_json}, "latest_only": true}}'
''')

    print("\nOption B - Using Python requests:")
    print("-" * 40)
    print(f'''
import requests

tickers = {unique_tickers}

# Split into batches of 50 for manageability
batch_size = 50
for i in range(0, len(tickers), batch_size):
    batch = tickers[i:i+batch_size]
    resp = requests.post(
        "http://localhost:8000/api/batch-analyze",
        json={{"tickers": batch, "latest_only": True}}
    )
    job_id = resp.json().get("job_id")
    print(f"Batch {{i//batch_size + 1}} started: job_id={{job_id}}")
''')

    print("\n" + "=" * 60)
    print("Ticker List Summary")
    print("=" * 60)
    print(f"\nUnique tickers ({len(unique_tickers)}):")
    for i in range(0, len(unique_tickers), 10):
        row = unique_tickers[i:i+10]
        print("  " + ", ".join(row))

    return unique_tickers


if __name__ == "__main__":
    main()
