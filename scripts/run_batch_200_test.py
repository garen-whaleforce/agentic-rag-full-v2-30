#!/usr/bin/env python3
"""
Batch test for 200 tickers with full output fields.
Outputs: ticker, date, year, quarter, category, expected_pred, predicted, confidence, actual, result
Reports summary every 10 minutes.
Saves final CSV with all columns.
"""

import sys
import os

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

import asyncio
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

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
REPORT_INTERVAL_SECONDS = 600  # 10 minutes

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
    "PWR", "KR", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "AMD", "QCOM",
    "MU", "INTC", "TXN", "AVGO", "CSCO", "ORCL", "CRM", "ADBE", "ACN", "SAP",
]

# Top 100 Losers (from PDF - 2025-01-01 to 2025-12-02)
TOP_LOSERS = [
    "FISV", "ALGN", "SNPS", "IT", "SWKS", "BAX", "UNH", "FTNT", "VRTX", "XYZ",
    "LULU", "ARE", "SRE", "HII", "SMCI", "CMG", "BDX", "LKQ", "DASH", "CI",
    "STZ", "COIN", "EME", "EL", "SJM", "ON", "NTAP", "NCLH", "VTRS", "ZBH",
    "COO", "GDDY", "LLY", "UPS", "AMAT", "ADBE", "ZTS", "BBY", "PYPL", "HRL",
    "IP", "NOC", "WDAY", "OTIS", "VST", "ELV", "SW", "HPE", "TDG", "ZBRA",
    "CPRT", "FIS", "IEX", "TMUS", "DVA", "CMCSA", "J", "LMT", "PAYC", "LYV",
    "VRSK", "FDS", "MDT", "ABT", "JNJ", "PFE", "MRK", "BMY", "GILD", "BIIB",
    "AMGN", "ABBV", "TMO", "DHR", "SYK", "BDX", "BSX", "EW", "ISRG", "ZBH",
    "HON", "MMM", "GE", "RTX", "LMT", "NOC", "BA", "CAT", "DE", "EMR",
    "ETN", "ITW", "PH", "ROK", "AME", "CTAS", "FAST", "PAYX", "ADP", "CDNS",
]

def get_unique_tickers(limit: int = 200) -> List[str]:
    """Remove duplicates while preserving order, limit to specified count."""
    seen = set()
    unique = []
    for t in TOP_GAINERS + TOP_LOSERS:
        if t not in seen and t != "XYZ":  # Skip invalid ticker XYZ
            seen.add(t)
            unique.append(t)
        if len(unique) >= limit:
            break
    return unique


def get_ticker_category(ticker: str) -> str:
    """Determine if ticker is GAINER or LOSER based on original list."""
    if ticker in TOP_GAINERS:
        return "GAINER"
    elif ticker in TOP_LOSERS:
        return "LOSER"
    return "UNKNOWN"


def get_expected_pred(category: str) -> str:
    """Expected prediction based on category."""
    if category == "GAINER":
        return "UP"
    elif category == "LOSER":
        return "DOWN"
    return "UNKNOWN"


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
            date_str = d.get("date", "")
            if y is None or q is None:
                continue
            try:
                valid.append((int(y), int(q), date_str))
            except Exception:
                continue
        if valid:
            valid.sort(reverse=True)
            return valid[0]  # (year, quarter, date)
    except Exception as e:
        print(f"  Warning: Could not get quarters for {ticker}: {e}")
    return None, None, None


# Global progress tracking
progress_lock = asyncio.Lock()
completed_count = 0
results_list: List[Dict] = []
start_time: float = 0
last_report_time: float = 0


def determine_result(predicted: str, actual: Optional[float]) -> str:
    """Determine if prediction was correct based on actual return."""
    if actual is None or predicted in ("N/A", "UNKNOWN", None):
        return "N/A"

    pred_upper = str(predicted).upper()
    if pred_upper == "UP":
        return "CORRECT" if actual > 0 else "WRONG"
    elif pred_upper == "DOWN":
        return "CORRECT" if actual < 0 else "WRONG"
    elif pred_upper == "NEUTRAL":
        return "CORRECT" if abs(actual) < 1.0 else "WRONG"
    return "N/A"


def print_summary_report(results: List[Dict], elapsed_seconds: float, is_final: bool = False):
    """Print a summary report of current progress."""
    report_type = "FINAL SUMMARY" if is_final else "PROGRESS REPORT"
    print("\n" + "=" * 70)
    print(f"{report_type} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed: {elapsed_seconds/60:.1f} minutes")
    print("=" * 70)

    total = len(results)
    success = [r for r in results if r.get("status") == "success"]
    errors = [r for r in results if r.get("status") != "success"]

    print(f"\nTotal processed: {total}")
    print(f"Success: {len(success)}")
    print(f"Errors: {len(errors)}")

    if success:
        # Prediction distribution
        up_pred = [r for r in success if r.get("predicted") == "UP"]
        down_pred = [r for r in success if r.get("predicted") == "DOWN"]
        neutral_pred = [r for r in success if r.get("predicted") == "NEUTRAL"]

        print(f"\nPrediction Distribution:")
        print(f"  UP: {len(up_pred)}")
        print(f"  DOWN: {len(down_pred)}")
        print(f"  NEUTRAL: {len(neutral_pred)}")

        # Accuracy by category
        gainers = [r for r in success if r.get("category") == "GAINER"]
        losers = [r for r in success if r.get("category") == "LOSER"]

        print(f"\nBy Category:")
        print(f"  GAINER: {len(gainers)}")
        print(f"  LOSER: {len(losers)}")

        # Result accuracy
        correct = [r for r in success if r.get("result") == "CORRECT"]
        wrong = [r for r in success if r.get("result") == "WRONG"]
        na_result = [r for r in success if r.get("result") == "N/A"]

        print(f"\nPrediction Accuracy:")
        print(f"  CORRECT: {len(correct)}")
        print(f"  WRONG: {len(wrong)}")
        print(f"  N/A: {len(na_result)}")

        if correct or wrong:
            accuracy = len(correct) / (len(correct) + len(wrong)) * 100
            print(f"  Accuracy: {accuracy:.1f}%")

        # Expected vs Predicted alignment
        aligned = [r for r in success if r.get("expected_pred") == r.get("predicted")]
        print(f"\nExpected vs Predicted Alignment: {len(aligned)}/{len(success)} ({len(aligned)/len(success)*100:.1f}%)")

        # High confidence signals
        high_conf_up = [r for r in up_pred if (r.get("confidence") or 0) >= 0.7]
        high_conf_down = [r for r in down_pred if (r.get("confidence") or 1) <= 0.3]

        print(f"\nHigh Confidence Signals:")
        print(f"  LONG (UP conf >= 0.7): {len(high_conf_up)}")
        print(f"  SHORT (DOWN conf <= 0.3): {len(high_conf_down)}")

    if errors and (is_final or len(errors) <= 5):
        print(f"\nErrors ({len(errors)}):")
        for r in errors[:10]:
            print(f"  {r['ticker']}: {r.get('status')} - {r.get('error', 'N/A')[:50]}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    print("=" * 70 + "\n")


async def report_progress_periodically(total_tickers: int):
    """Background task to report progress every 10 minutes."""
    global last_report_time, results_list, start_time

    while True:
        await asyncio.sleep(60)  # Check every minute

        current_time = time.time()
        if current_time - last_report_time >= REPORT_INTERVAL_SECONDS:
            async with progress_lock:
                if results_list:
                    elapsed = current_time - start_time
                    print(f"\n[Progress: {len(results_list)}/{total_tickers}]")
                    print_summary_report(results_list.copy(), elapsed, is_final=False)
                    last_report_time = current_time


async def analyze_single(ticker: str, idx: int, total: int, semaphore: asyncio.Semaphore) -> Dict:
    """Analyze a single ticker with semaphore for concurrency control."""
    global completed_count, results_list

    async with semaphore:
        analysis_start = time.time()
        category = get_ticker_category(ticker)
        expected_pred = get_expected_pred(category)

        print(f"\n[{idx}/{total}] Analyzing {ticker} ({category})...")

        result_row = {
            "ticker": ticker,
            "date": "",
            "year": None,
            "quarter": None,
            "category": category,
            "expected_pred": expected_pred,
            "predicted": "N/A",
            "confidence": None,
            "actual": None,
            "result": "N/A",
            "status": "pending",
            "error": None,
            "elapsed": 0,
        }

        try:
            quarter_info = get_latest_quarter(ticker)
            if not quarter_info or quarter_info == (None, None, None):
                result_row["status"] = "NO_QUARTERS"
                result_row["error"] = "No available quarters found"
                result_row["elapsed"] = time.time() - analysis_start
                async with progress_lock:
                    results_list.append(result_row)
                return result_row

            year, quarter, date_str = quarter_info
            if not year or not quarter:
                result_row["status"] = "NO_QUARTERS"
                result_row["error"] = "No available quarters found"
                result_row["elapsed"] = time.time() - analysis_start
                async with progress_lock:
                    results_list.append(result_row)
                return result_row

            result_row["year"] = year
            result_row["quarter"] = quarter
            result_row["date"] = date_str

            print(f"  {ticker}: {year}-Q{quarter} ({date_str})")

            analysis_result = await analyze_earnings_async(ticker, year, quarter, skip_cache=True)
            elapsed = time.time() - analysis_start
            result_row["elapsed"] = elapsed

            if analysis_result and "error" not in analysis_result:
                # Extract prediction and confidence from agentic_result
                agentic_result = analysis_result.get("agentic_result", {})
                if isinstance(agentic_result, dict):
                    result_row["predicted"] = agentic_result.get("prediction", "N/A")
                    result_row["confidence"] = agentic_result.get("confidence")

                # Get actual return from backtest or post_earnings_return
                backtest = analysis_result.get("backtest", {})
                if backtest and isinstance(backtest, dict):
                    result_row["actual"] = backtest.get("change_pct")
                if result_row["actual"] is None:
                    result_row["actual"] = analysis_result.get("post_earnings_return")

                # Determine if prediction was correct
                result_row["result"] = determine_result(
                    result_row["predicted"],
                    result_row["actual"]
                )
                result_row["status"] = "success"

                print(f"  {ticker}: {result_row['predicted']} (conf={result_row['confidence']}) "
                      f"actual={result_row['actual']}% -> {result_row['result']} [{elapsed:.1f}s]")
            else:
                error_msg = analysis_result.get("error", "Unknown error") if analysis_result else "No result"
                result_row["status"] = "ERROR"
                result_row["error"] = error_msg
                print(f"  {ticker}: ERROR - {error_msg[:50]} [{elapsed:.1f}s]")

        except Exception as e:
            elapsed = time.time() - analysis_start
            result_row["status"] = "EXCEPTION"
            result_row["error"] = str(e)
            result_row["elapsed"] = elapsed
            print(f"  {ticker}: EXCEPTION - {str(e)[:50]} [{elapsed:.1f}s]")

        async with progress_lock:
            results_list.append(result_row)

        return result_row


def save_results_csv(results: List[Dict], output_file: Path):
    """Save results to CSV file."""
    fieldnames = [
        "ticker", "date", "year", "quarter", "category",
        "expected_pred", "predicted", "confidence", "actual", "result",
        "status", "error", "elapsed"
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nCSV saved to: {output_file}")


async def main():
    global start_time, last_report_time, results_list

    print("=" * 70)
    print("BATCH TEST - 200 TICKERS")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Report interval: {REPORT_INTERVAL_SECONDS // 60} minutes")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Apply profile
    if not apply_profile("garen1204"):
        print("Warning: Could not apply profile, continuing with defaults...")

    tickers = get_unique_tickers(200)
    total = len(tickers)
    print(f"\nTotal unique tickers: {total}")

    # Initialize timing
    start_time = time.time()
    last_report_time = start_time
    results_list = []

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(CONCURRENCY)

    # Start progress reporter
    reporter_task = asyncio.create_task(report_progress_periodically(total))

    # Create all tasks
    tasks = [
        analyze_single(ticker, idx, total, semaphore)
        for idx, ticker in enumerate(tickers, 1)
    ]

    # Run all tasks concurrently (semaphore limits actual concurrency)
    print(f"\nRunning {total} analyses with concurrency={CONCURRENCY}...\n")

    try:
        await asyncio.gather(*tasks)
    finally:
        reporter_task.cancel()
        try:
            await reporter_task
        except asyncio.CancelledError:
            pass

    # Final summary
    elapsed = time.time() - start_time
    print_summary_report(results_list, elapsed, is_final=True)

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = Path(__file__).parent / f"batch_results_{timestamp}.csv"
    save_results_csv(results_list, csv_file)

    # Also save as JSON for detailed analysis
    json_file = Path(__file__).parent / f"batch_results_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(results_list, f, indent=2, default=str)
    print(f"JSON saved to: {json_file}")

    print(f"\nTotal elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Average per ticker: {elapsed/total:.1f}s")
    print(f"Effective time (with concurrency {CONCURRENCY}): ~{elapsed:.1f}s")

    return results_list


if __name__ == "__main__":
    asyncio.run(main())
