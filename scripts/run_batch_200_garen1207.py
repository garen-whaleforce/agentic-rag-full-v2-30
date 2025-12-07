#!/usr/bin/env python3
"""
Batch test for 200 tickers using garen1207 profile.
- No cache (skip_cache=True)
- Summary report every 10 minutes
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
from fmp_client import get_transcript_dates

CONCURRENCY = 5
REPORT_INTERVAL_SECONDS = 600  # 10 minutes
PROFILE_NAME = "garen1207"

# Top 100 Gainers from PDF (2025-01-01 to 2025-12-02)
TOP_GAINERS = [
    {"ticker": "ORCL", "date": "2025-09-09", "change": 35.95},
    {"ticker": "IDXX", "date": "2025-08-04", "change": 27.49},
    {"ticker": "NRG", "date": "2025-05-12", "change": 26.21},
    {"ticker": "APP", "date": "2025-02-12", "change": 24.02},
    {"ticker": "PLTR", "date": "2025-02-03", "change": 23.99},
    {"ticker": "DAL", "date": "2025-04-09", "change": 23.38},
    {"ticker": "DDOG", "date": "2025-11-06", "change": 23.13},
    {"ticker": "WST", "date": "2025-07-24", "change": 22.78},
    {"ticker": "JBHT", "date": "2025-10-15", "change": 22.14},
    {"ticker": "PODD", "date": "2025-05-08", "change": 20.88},
    {"ticker": "CHRW", "date": "2025-10-29", "change": 19.71},
    {"ticker": "GNRC", "date": "2025-07-30", "change": 19.61},
    {"ticker": "STX", "date": "2025-10-28", "change": 19.11},
    {"ticker": "CRL", "date": "2025-05-07", "change": 18.68},
    {"ticker": "TTD", "date": "2025-05-08", "change": 18.60},
    {"ticker": "EBAY", "date": "2025-07-30", "change": 18.30},
    {"ticker": "CHRW", "date": "2025-07-30", "change": 18.10},
    {"ticker": "IQV", "date": "2025-07-22", "change": 17.88},
    {"ticker": "EXPE", "date": "2025-11-06", "change": 17.55},
    {"ticker": "ANET", "date": "2025-08-05", "change": 17.49},
    {"ticker": "MGM", "date": "2025-02-12", "change": 17.46},
    {"ticker": "EXPE", "date": "2025-02-06", "change": 17.27},
    {"ticker": "DHI", "date": "2025-07-22", "change": 16.98},
    {"ticker": "AXON", "date": "2025-08-04", "change": 16.41},
    {"ticker": "LW", "date": "2025-07-23", "change": 16.31},
    {"ticker": "DXCM", "date": "2025-05-01", "change": 16.17},
    {"ticker": "DG", "date": "2025-06-03", "change": 15.85},
    {"ticker": "NOW", "date": "2025-04-23", "change": 15.49},
    {"ticker": "CAH", "date": "2025-10-30", "change": 15.43},
    {"ticker": "AXON", "date": "2025-02-25", "change": 15.25},
    {"ticker": "NKE", "date": "2025-06-26", "change": 15.19},
    {"ticker": "CVS", "date": "2025-02-12", "change": 14.95},
    {"ticker": "GM", "date": "2025-10-21", "change": 14.86},
    {"ticker": "IDXX", "date": "2025-11-03", "change": 14.84},
    {"ticker": "AKAM", "date": "2025-11-06", "change": 14.71},
    {"ticker": "GEV", "date": "2025-07-23", "change": 14.58},
    {"ticker": "HAS", "date": "2025-04-24", "change": 14.58},
    {"ticker": "ABNB", "date": "2025-02-13", "change": 14.45},
    {"ticker": "FSLR", "date": "2025-10-30", "change": 14.28},
    {"ticker": "AXON", "date": "2025-05-07", "change": 14.13},
    {"ticker": "HOOD", "date": "2025-02-12", "change": 14.11},
    {"ticker": "TTWO", "date": "2025-02-06", "change": 14.03},
    {"ticker": "ISRG", "date": "2025-10-21", "change": 13.89},
    {"ticker": "EFX", "date": "2025-04-22", "change": 13.84},
    {"ticker": "ULTA", "date": "2025-03-13", "change": 13.68},
    {"ticker": "ORCL", "date": "2025-06-11", "change": 13.31},
    {"ticker": "IBM", "date": "2025-01-29", "change": 12.96},
    {"ticker": "HAS", "date": "2025-02-20", "change": 12.95},
    {"ticker": "DOW", "date": "2025-10-23", "change": 12.95},
    {"ticker": "EPAM", "date": "2025-05-08", "change": 12.88},
    {"ticker": "ROK", "date": "2025-02-10", "change": 12.65},
    {"ticker": "GRMN", "date": "2025-02-19", "change": 12.64},
    {"ticker": "MCHP", "date": "2025-05-08", "change": 12.60},
    {"ticker": "INTU", "date": "2025-02-25", "change": 12.58},
    {"ticker": "CNC", "date": "2025-10-29", "change": 12.50},
    {"ticker": "HUM", "date": "2025-07-30", "change": 12.40},
    {"ticker": "LVS", "date": "2025-10-22", "change": 12.39},
    {"ticker": "KVUE", "date": "2025-11-03", "change": 12.32},
    {"ticker": "F", "date": "2025-10-23", "change": 12.16},
    {"ticker": "TPR", "date": "2025-02-06", "change": 12.02},
    {"ticker": "RCL", "date": "2025-01-28", "change": 12.00},
    {"ticker": "DAL", "date": "2025-07-10", "change": 11.99},
    {"ticker": "APP", "date": "2025-08-06", "change": 11.97},
    {"ticker": "TEL", "date": "2025-07-23", "change": 11.95},
    {"ticker": "ROK", "date": "2025-05-07", "change": 11.90},
    {"ticker": "APP", "date": "2025-05-07", "change": 11.88},
    {"ticker": "GLW", "date": "2025-07-29", "change": 11.86},
    {"ticker": "REGN", "date": "2025-10-28", "change": 11.82},
    {"ticker": "ULTA", "date": "2025-05-29", "change": 11.78},
    {"ticker": "AES", "date": "2025-02-28", "change": 11.66},
    {"ticker": "CAT", "date": "2025-10-29", "change": 11.63},
    {"ticker": "CARR", "date": "2025-05-01", "change": 11.61},
    {"ticker": "HAL", "date": "2025-10-21", "change": 11.58},
    {"ticker": "STX", "date": "2025-04-29", "change": 11.56},
    {"ticker": "PHM", "date": "2025-07-22", "change": 11.52},
    {"ticker": "CHTR", "date": "2025-04-25", "change": 11.43},
    {"ticker": "FFIV", "date": "2025-01-28", "change": 11.40},
    {"ticker": "DECK", "date": "2025-07-24", "change": 11.35},
    {"ticker": "JCI", "date": "2025-02-05", "change": 11.28},
    {"ticker": "META", "date": "2025-07-30", "change": 11.25},
    {"ticker": "IDXX", "date": "2025-02-03", "change": 11.13},
    {"ticker": "LVS", "date": "2025-01-29", "change": 11.08},
    {"ticker": "PM", "date": "2025-02-06", "change": 10.95},
    {"ticker": "WST", "date": "2025-10-23", "change": 10.92},
    {"ticker": "EXPD", "date": "2025-11-04", "change": 10.84},
    {"ticker": "HSIC", "date": "2025-11-04", "change": 10.78},
    {"ticker": "DIS", "date": "2025-05-07", "change": 10.76},
    {"ticker": "XYL", "date": "2025-07-31", "change": 10.74},
    {"ticker": "NRG", "date": "2025-02-26", "change": 10.63},
    {"ticker": "MPWR", "date": "2025-07-31", "change": 10.46},
    {"ticker": "WYNN", "date": "2025-02-13", "change": 10.38},
    {"ticker": "BEN", "date": "2025-01-31", "change": 10.37},
    {"ticker": "ALLE", "date": "2025-04-24", "change": 10.32},
    {"ticker": "CEG", "date": "2025-05-06", "change": 10.29},
    {"ticker": "INCY", "date": "2025-07-29", "change": 10.29},
    {"ticker": "WDC", "date": "2025-07-30", "change": 10.16},
    {"ticker": "KEYS", "date": "2025-11-24", "change": 10.01},
    {"ticker": "LW", "date": "2025-04-03", "change": 10.01},
    {"ticker": "PWR", "date": "2025-05-01", "change": 9.99},
    {"ticker": "KR", "date": "2025-06-20", "change": 9.84},
]

# Top 100 Losers from PDF (2025-01-01 to 2025-12-02)
TOP_LOSERS = [
    {"ticker": "FISV", "date": "2025-10-29", "change": -44.04},
    {"ticker": "TTD", "date": "2025-08-07", "change": -38.61},
    {"ticker": "WST", "date": "2025-02-13", "change": -38.22},
    {"ticker": "ALGN", "date": "2025-07-30", "change": -36.63},
    {"ticker": "SNPS", "date": "2025-09-09", "change": -35.84},
    {"ticker": "TTD", "date": "2025-02-12", "change": -32.98},
    {"ticker": "IT", "date": "2025-08-05", "change": -27.55},
    {"ticker": "SWKS", "date": "2025-02-05", "change": -24.67},
    {"ticker": "BAX", "date": "2025-07-31", "change": -22.42},
    {"ticker": "UNH", "date": "2025-04-17", "change": -22.38},
    {"ticker": "FTNT", "date": "2025-08-06", "change": -22.03},
    {"ticker": "AKAM", "date": "2025-02-20", "change": -21.73},
    {"ticker": "VRTX", "date": "2025-08-04", "change": -20.60},
    {"ticker": "DECK", "date": "2025-01-30", "change": -20.51},
    {"ticker": "XYZ", "date": "2025-05-01", "change": -20.43},
    {"ticker": "DECK", "date": "2025-05-22", "change": -19.86},
    {"ticker": "LULU", "date": "2025-06-05", "change": -19.80},
    {"ticker": "ARE", "date": "2025-10-27", "change": -19.17},
    {"ticker": "SRE", "date": "2025-02-25", "change": -18.97},
    {"ticker": "LULU", "date": "2025-09-04", "change": -18.58},
    {"ticker": "FISV", "date": "2025-04-24", "change": -18.53},
    {"ticker": "CHTR", "date": "2025-07-25", "change": -18.49},
    {"ticker": "HII", "date": "2025-02-06", "change": -18.32},
    {"ticker": "SMCI", "date": "2025-08-05", "change": -18.29},
    {"ticker": "CMG", "date": "2025-10-29", "change": -18.18},
    {"ticker": "BDX", "date": "2025-05-01", "change": -18.13},
    {"ticker": "LKQ", "date": "2025-07-24", "change": -17.82},
    {"ticker": "XYZ", "date": "2025-02-20", "change": -17.69},
    {"ticker": "DASH", "date": "2025-11-05", "change": -17.45},
    {"ticker": "DOW", "date": "2025-07-24", "change": -17.45},
    {"ticker": "CI", "date": "2025-10-30", "change": -17.39},
    {"ticker": "STZ", "date": "2025-01-10", "change": -17.09},
    {"ticker": "COIN", "date": "2025-07-31", "change": -16.70},
    {"ticker": "EME", "date": "2025-10-30", "change": -16.60},
    {"ticker": "EL", "date": "2025-02-04", "change": -16.07},
    {"ticker": "EBAY", "date": "2025-10-29", "change": -15.88},
    {"ticker": "TPR", "date": "2025-08-14", "change": -15.71},
    {"ticker": "SJM", "date": "2025-06-10", "change": -15.59},
    {"ticker": "ON", "date": "2025-08-04", "change": -15.58},
    {"ticker": "NTAP", "date": "2025-02-27", "change": -15.57},
    {"ticker": "NCLH", "date": "2025-11-04", "change": -15.28},
    {"ticker": "DECK", "date": "2025-10-23", "change": -15.21},
    {"ticker": "VTRS", "date": "2025-02-27", "change": -15.21},
    {"ticker": "ZBH", "date": "2025-11-05", "change": -15.15},
    {"ticker": "DXCM", "date": "2025-10-30", "change": -14.63},
    {"ticker": "COO", "date": "2025-05-29", "change": -14.61},
    {"ticker": "BAX", "date": "2025-10-30", "change": -14.54},
    {"ticker": "GDDY", "date": "2025-02-13", "change": -14.28},
    {"ticker": "LULU", "date": "2025-03-27", "change": -14.19},
    {"ticker": "LLY", "date": "2025-08-07", "change": -14.14},
    {"ticker": "UPS", "date": "2025-01-30", "change": -14.11},
    {"ticker": "AMAT", "date": "2025-08-14", "change": -14.07},
    {"ticker": "ADBE", "date": "2025-03-12", "change": -13.85},
    {"ticker": "FISV", "date": "2025-07-23", "change": -13.84},
    {"ticker": "ZTS", "date": "2025-11-04", "change": -13.78},
    {"ticker": "NRG", "date": "2025-08-06", "change": -13.61},
    {"ticker": "TXN", "date": "2025-07-22", "change": -13.34},
    {"ticker": "CMG", "date": "2025-07-23", "change": -13.34},
    {"ticker": "BBY", "date": "2025-03-04", "change": -13.30},
    {"ticker": "PYPL", "date": "2025-02-04", "change": -13.17},
    {"ticker": "HRL", "date": "2025-08-28", "change": -13.09},
    {"ticker": "COO", "date": "2025-08-27", "change": -12.86},
    {"ticker": "IP", "date": "2025-07-31", "change": -12.85},
    {"ticker": "EPAM", "date": "2025-02-20", "change": -12.80},
    {"ticker": "IP", "date": "2025-10-30", "change": -12.66},
    {"ticker": "NOC", "date": "2025-04-22", "change": -12.66},
    {"ticker": "WDAY", "date": "2025-05-22", "change": -12.52},
    {"ticker": "OTIS", "date": "2025-07-23", "change": -12.38},
    {"ticker": "VST", "date": "2025-02-27", "change": -12.27},
    {"ticker": "ELV", "date": "2025-07-17", "change": -12.22},
    {"ticker": "SW", "date": "2025-10-29", "change": -12.18},
    {"ticker": "PLTR", "date": "2025-05-05", "change": -12.05},
    {"ticker": "HPE", "date": "2025-03-06", "change": -11.97},
    {"ticker": "TDG", "date": "2025-08-05", "change": -11.94},
    {"ticker": "ZBRA", "date": "2025-10-28", "change": -11.68},
    {"ticker": "LLY", "date": "2025-05-01", "change": -11.66},
    {"ticker": "ZBH", "date": "2025-05-05", "change": -11.62},
    {"ticker": "LKQ", "date": "2025-04-24", "change": -11.56},
    {"ticker": "CPRT", "date": "2025-05-22", "change": -11.52},
    {"ticker": "FIS", "date": "2025-02-11", "change": -11.49},
    {"ticker": "GRMN", "date": "2025-10-29", "change": -11.48},
    {"ticker": "NOW", "date": "2025-01-29", "change": -11.44},
    {"ticker": "ZBRA", "date": "2025-08-05", "change": -11.35},
    {"ticker": "META", "date": "2025-10-29", "change": -11.33},
    {"ticker": "SMCI", "date": "2025-11-04", "change": -11.33},
    {"ticker": "IEX", "date": "2025-07-30", "change": -11.29},
    {"ticker": "GDDY", "date": "2025-08-07", "change": -11.25},
    {"ticker": "TMUS", "date": "2025-04-24", "change": -11.22},
    {"ticker": "DVA", "date": "2025-02-13", "change": -11.09},
    {"ticker": "CMCSA", "date": "2025-01-30", "change": -11.00},
    {"ticker": "J", "date": "2025-11-20", "change": -10.95},
    {"ticker": "LMT", "date": "2025-07-22", "change": -10.81},
    {"ticker": "HOOD", "date": "2025-11-05", "change": -10.81},
    {"ticker": "AKAM", "date": "2025-05-08", "change": -10.76},
    {"ticker": "PAYC", "date": "2025-11-05", "change": -10.72},
    {"ticker": "CARR", "date": "2025-07-29", "change": -10.61},
    {"ticker": "LYV", "date": "2025-11-04", "change": -10.59},
    {"ticker": "UPS", "date": "2025-07-29", "change": -10.57},
    {"ticker": "VRSK", "date": "2025-10-29", "change": -10.39},
    {"ticker": "FDS", "date": "2025-09-18", "change": -10.36},
]


def get_test_list() -> List[Dict]:
    """Combine gainers and losers into test list."""
    test_list = []
    for i, g in enumerate(TOP_GAINERS, 1):
        test_list.append({
            "rank": i,
            "ticker": g["ticker"],
            "earnings_date": g["date"],
            "category": "GAINER",
            "expected_pred": "UP",
            "actual_change": g["change"],
        })
    for i, l in enumerate(TOP_LOSERS, 1):
        test_list.append({
            "rank": 100 + i,
            "ticker": l["ticker"],
            "earnings_date": l["date"],
            "category": "LOSER",
            "expected_pred": "DOWN",
            "actual_change": l["change"],
        })
    return test_list


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


def date_to_fiscal_quarter(date_str: str):
    """Convert date to fiscal year and quarter."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        quarter = (dt.month - 1) // 3 + 1
        return dt.year, quarter
    except:
        return None, None


def get_quarter_for_date(ticker: str, target_date: str):
    """Get year/quarter for a specific earnings date."""
    try:
        dates = get_transcript_dates(ticker)
        for d in dates:
            d_date = d.get("date", "")[:10]
            if d_date == target_date:
                y = d.get("year") or d.get("calendar_year")
                q = d.get("quarter") or d.get("calendar_quarter")
                if y and q:
                    return int(y), int(q), d_date
        # Fallback: infer from date
        y, q = date_to_fiscal_quarter(target_date)
        if y and q:
            return y, q, target_date
    except Exception as e:
        print(f"  Warning: Could not get quarter for {ticker}: {e}", flush=True)
    return None, None, target_date


# Global tracking
progress_lock = asyncio.Lock()
results_list: List[Dict] = []
start_time: float = 0
last_report_time: float = 0


def determine_result(predicted: str, actual: Optional[float]) -> str:
    """Determine if prediction was correct."""
    if actual is None or predicted in ("N/A", "UNKNOWN", None, ""):
        return "N/A"
    pred_upper = str(predicted).upper()
    if pred_upper == "UP":
        return "CORRECT" if actual > 0 else "WRONG"
    elif pred_upper == "DOWN":
        return "CORRECT" if actual < 0 else "WRONG"
    elif pred_upper == "NEUTRAL":
        return "SKIP"
    return "N/A"


def extract_short_term_tag(note_text: str) -> str:
    """Extract ShortTermTag from helper note."""
    if not note_text:
        return "N/A"
    import re
    match = re.search(r"ShortTermTag:\s*(Bullish|Bearish|Neutral|Unclear)", note_text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return "N/A"


def print_summary_report(results: List[Dict], elapsed_seconds: float, is_final: bool = False):
    """Print a summary report in table format."""
    report_type = "FINAL SUMMARY" if is_final else "10-MIN PROGRESS REPORT"
    print(f"\n{'='*140}", flush=True)
    print(f"{report_type} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Profile: {PROFILE_NAME}", flush=True)
    print(f"Elapsed: {elapsed_seconds/60:.1f} minutes", flush=True)
    print(f"{'='*140}", flush=True)

    total = len(results)
    success = [r for r in results if r.get("status") == "success"]
    errors = [r for r in results if r.get("status") != "success"]

    print(f"\nTotal processed: {total}/200", flush=True)
    print(f"Success: {len(success)}", flush=True)
    print(f"Errors: {len(errors)}", flush=True)

    if success:
        up_pred = [r for r in success if r.get("predicted") == "UP"]
        down_pred = [r for r in success if r.get("predicted") == "DOWN"]
        neutral_pred = [r for r in success if r.get("predicted") == "NEUTRAL"]

        print(f"\nPrediction Distribution: UP={len(up_pred)}, DOWN={len(down_pred)}, NEUTRAL={len(neutral_pred)}", flush=True)

        correct = [r for r in success if r.get("result") == "CORRECT"]
        wrong = [r for r in success if r.get("result") == "WRONG"]
        skip = [r for r in success if r.get("result") == "SKIP"]

        print(f"Results: CORRECT={len(correct)}, WRONG={len(wrong)}, SKIP(NEUTRAL)={len(skip)}", flush=True)
        trades = len(correct) + len(wrong)
        if trades > 0:
            acc = len(correct) / trades * 100
            print(f"Trading Accuracy (excl. NEUTRAL): {acc:.1f}% ({len(correct)}/{trades})", flush=True)

        # Print table header
        print(f"\n{'rank':<5} {'ticker':<6} {'year':<5} {'quarter':<8} {'date':<12} {'category':<9} {'expected':<9} {'predicted':<10} {'confidence':<11} {'actual':<10} {'result':<8}", flush=True)
        print("-" * 140, flush=True)

        # Show last 15 results
        for r in results[-15:]:
            if r.get("status") == "success":
                conf_str = f"{r.get('confidence', 0):.1f}" if r.get('confidence') is not None else "N/A"
                actual_str = f"{r.get('actual', 0):+.2f}%" if r.get('actual') is not None else "N/A"
                print(f"{r.get('rank', '-'):<5} {r.get('ticker', '-'):<6} {r.get('year', '-'):<5} Q{r.get('quarter', '-'):<7} {r.get('date', '-'):<12} {r.get('category', '-'):<9} {r.get('expected_pred', '-'):<9} {r.get('predicted', '-'):<10} {conf_str:<11} {actual_str:<10} {r.get('result', '-'):<8}", flush=True)

    print(f"{'='*140}\n", flush=True)


async def report_progress_periodically(total_tickers: int):
    """Background task to report progress every 10 minutes."""
    global last_report_time, results_list, start_time
    while True:
        await asyncio.sleep(60)
        current_time = time.time()
        if current_time - last_report_time >= REPORT_INTERVAL_SECONDS:
            async with progress_lock:
                if results_list:
                    elapsed = current_time - start_time
                    print_summary_report(results_list.copy(), elapsed, is_final=False)
                    last_report_time = current_time


async def analyze_single(item: Dict, idx: int, total: int, semaphore: asyncio.Semaphore) -> Dict:
    """Analyze a single ticker."""
    global results_list

    async with semaphore:
        analysis_start = time.time()
        ticker = item["ticker"]
        category = item["category"]
        expected_pred = item["expected_pred"]
        target_date = item["earnings_date"]
        actual_from_pdf = item["actual_change"]

        print(f"\n[{idx}/{total}] {ticker} ({category}) date={target_date}...", flush=True)

        result_row = {
            "rank": item["rank"],
            "ticker": ticker,
            "date": target_date,
            "year": None,
            "quarter": None,
            "category": category,
            "expected_pred": expected_pred,
            "predicted": "N/A",
            "confidence": None,
            "actual": actual_from_pdf,
            "result": "N/A",
            "status": "pending",
            "error": None,
            "elapsed": 0,
            # Detailed agent outputs
            "main_summary": None,
            "notes_financials": None,
            "notes_past": None,
            "notes_peers": None,
            "tag_financials": None,
            "tag_past": None,
            "tag_peers": None,
        }

        try:
            year, quarter, date_str = get_quarter_for_date(ticker, target_date)
            if not year or not quarter:
                result_row["status"] = "NO_QUARTERS"
                result_row["error"] = "Could not determine quarter"
                result_row["elapsed"] = time.time() - analysis_start
                async with progress_lock:
                    results_list.append(result_row)
                print(f"  {ticker}: NO_QUARTERS", flush=True)
                return result_row

            result_row["year"] = year
            result_row["quarter"] = quarter
            result_row["date"] = date_str

            print(f"  {ticker}: {year}-Q{quarter}", flush=True)

            analysis_result = await analyze_earnings_async(ticker, year, quarter, skip_cache=True)
            elapsed = time.time() - analysis_start
            result_row["elapsed"] = elapsed

            if analysis_result and "error" not in analysis_result:
                agentic_result = analysis_result.get("agentic_result", {})
                if isinstance(agentic_result, dict):
                    result_row["predicted"] = agentic_result.get("prediction", "N/A")
                    result_row["confidence"] = agentic_result.get("confidence")
                    result_row["main_summary"] = agentic_result.get("summary", "")

                    # Extract detailed notes from raw
                    raw = agentic_result.get("raw", {})
                    if isinstance(raw, dict):
                        notes = raw.get("notes", {})
                        if isinstance(notes, dict):
                            result_row["notes_financials"] = notes.get("financials", "")
                            result_row["notes_past"] = notes.get("past", "")
                            result_row["notes_peers"] = notes.get("peers", "")
                            # Extract ShortTermTags
                            result_row["tag_financials"] = extract_short_term_tag(notes.get("financials", ""))
                            result_row["tag_past"] = extract_short_term_tag(notes.get("past", ""))
                            result_row["tag_peers"] = extract_short_term_tag(notes.get("peers", ""))

                result_row["result"] = determine_result(result_row["predicted"], actual_from_pdf)
                result_row["status"] = "success"

                tags_str = f"[F:{result_row['tag_financials']} P:{result_row['tag_past']} C:{result_row['tag_peers']}]"
                print(f"  {ticker}: {result_row['predicted']} (conf={result_row['confidence']}) {tags_str} actual={actual_from_pdf}% -> {result_row['result']} [{elapsed:.0f}s]", flush=True)
            else:
                error_msg = analysis_result.get("error", "Unknown") if analysis_result else "No result"
                result_row["status"] = "ERROR"
                result_row["error"] = error_msg[:200]
                print(f"  {ticker}: ERROR - {error_msg[:80]} [{elapsed:.0f}s]", flush=True)

        except Exception as e:
            elapsed = time.time() - analysis_start
            result_row["status"] = "EXCEPTION"
            result_row["error"] = str(e)[:200]
            result_row["elapsed"] = elapsed
            print(f"  {ticker}: EXCEPTION - {str(e)[:80]} [{elapsed:.0f}s]", flush=True)

        async with progress_lock:
            results_list.append(result_row)

        return result_row


def save_results_csv(results: List[Dict], output_file: Path):
    """Save results to CSV file with all agent outputs."""
    fieldnames = [
        "rank", "ticker", "date", "year", "quarter", "category",
        "expected_pred", "predicted", "confidence", "actual", "result",
        "tag_financials", "tag_past", "tag_peers",
        "status", "error", "elapsed",
        "main_summary", "notes_financials", "notes_past", "notes_peers"
    ]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # Truncate long text fields for CSV readability
            write_row = row.copy()
            for field in ["main_summary", "notes_financials", "notes_past", "notes_peers"]:
                if write_row.get(field) and len(str(write_row[field])) > 5000:
                    write_row[field] = str(write_row[field])[:5000] + "...[truncated]"
            writer.writerow(write_row)
    print(f"\nCSV saved to: {output_file}", flush=True)


async def main():
    global start_time, last_report_time, results_list

    print("=" * 140, flush=True)
    print(f"BATCH TEST - 200 TICKERS using {PROFILE_NAME}", flush=True)
    print(f"Concurrency: {CONCURRENCY}", flush=True)
    print(f"Report interval: {REPORT_INTERVAL_SECONDS // 60} minutes", flush=True)
    print(f"Cache: DISABLED (skip_cache=True)", flush=True)
    print(f"Started: {datetime.now().isoformat()}", flush=True)
    print("=" * 140, flush=True)

    if not apply_profile(PROFILE_NAME):
        print(f"Failed to apply profile {PROFILE_NAME}, exiting.", flush=True)
        return []

    test_list = get_test_list()
    total = len(test_list)
    print(f"\nTotal tickers: {total} (100 gainers + 100 losers)", flush=True)

    start_time = time.time()
    last_report_time = start_time
    results_list = []

    semaphore = asyncio.Semaphore(CONCURRENCY)
    reporter_task = asyncio.create_task(report_progress_periodically(total))

    tasks = [
        analyze_single(item, idx, total, semaphore)
        for idx, item in enumerate(test_list, 1)
    ]

    print(f"\nRunning {total} analyses with concurrency={CONCURRENCY}...\n", flush=True)

    try:
        await asyncio.gather(*tasks)
    finally:
        reporter_task.cancel()
        try:
            await reporter_task
        except asyncio.CancelledError:
            pass

    elapsed = time.time() - start_time
    print_summary_report(results_list, elapsed, is_final=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = Path(__file__).parent / f"batch_garen1207_{timestamp}.csv"
    save_results_csv(results_list, csv_file)

    json_file = Path(__file__).parent / f"batch_garen1207_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(results_list, f, indent=2, default=str)
    print(f"JSON saved to: {json_file}", flush=True)

    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"Average per ticker: {elapsed/total:.1f}s", flush=True)

    return results_list


if __name__ == "__main__":
    asyncio.run(main())
