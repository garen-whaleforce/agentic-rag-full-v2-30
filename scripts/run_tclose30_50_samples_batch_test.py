#!/usr/bin/env python3
"""
Batch Test Script - Top 25 Gainers + Top 25 Losers from AWS DB

Fetches T+30 top gainers/losers directly from AWS PostgreSQL database,
runs analysis on each, and outputs results to CSV.

T+30 Calculation:
- BMO (Before Market Open): T+30 price vs T price (earnings day close)
- AMC (After Market Close): T+30 price vs T+1 price (next trading day close)

Usage:
    python scripts/run_batch_50_from_db.py
"""

import asyncio
import csv
import json
import sys
import time
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
REPORT_INTERVAL_SECONDS = 60  # 1 minute for smaller batch
TOP_N = 25  # Top 25 gainers + Top 25 losers = 50 total
DATE_RANGE_START = "2024-01-01"
DATE_RANGE_END = "2026-01-01"

# Type definitions
EarningsSession = Literal["BMO", "AMC", "UNKNOWN"]


def get_session_from_db(symbol: str, year: int, quarter: int) -> EarningsSession:
    """
    Get BMO/AMC session from AWS DB market_timing field.

    AWS DB values: 'before_market', 'after_market', 'undetermine'

    Returns: 'BMO', 'AMC', or 'UNKNOWN'
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
    # Fetch a range around target date to handle weekends/holidays
    try:
        dt = datetime.strptime(target_date, "%Y-%m-%d")
        start = (dt - timedelta(days=5)).strftime("%Y-%m-%d")
        end = (dt + timedelta(days=5)).strftime("%Y-%m-%d")

        prices = get_historical_prices(symbol, start, end)
        if not prices:
            return None

        # Find exact date match
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

        # Find first trading day after the target date
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
        # Fetch enough days to cover N trading days (account for weekends/holidays)
        end = (dt + timedelta(days=n + 20)).strftime("%Y-%m-%d")

        prices = get_historical_prices(symbol, base_date, end)
        if not prices:
            return None, None

        # Count trading days from base_date
        trading_days_after = [p for p in prices if p.get("date") > base_date]

        if len(trading_days_after) >= n:
            target = trading_days_after[n - 1]  # n-1 because list is 0-indexed
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

    Returns dict with:
    - session: BMO/AMC/UNKNOWN
    - base_date: the date used as baseline (T for BMO, T+1 for AMC)
    - base_price: the baseline price
    - t30_date: T+30 trading day date
    - t30_price: T+30 price
    - t30_change: percentage change
    """
    result = {
        "session": "UNKNOWN",
        "base_date": None,
        "base_price": None,
        "t30_date": None,
        "t30_price": None,
        "t30_change": None,
    }

    # Get session from AWS DB market_timing field
    result["session"] = get_session_from_db(symbol, year, quarter)

    # Determine base date based on session
    if result["session"] == "BMO":
        # BMO: base is T (earnings day close)
        base_date = earnings_date
        base_price = get_trading_day_price(symbol, earnings_date)

        # If earnings_date is not a trading day, find the closest one
        if base_price is None:
            next_date, next_price = get_next_trading_day_price(symbol, earnings_date)
            if next_date:
                base_date = next_date
                base_price = next_price
    else:
        # AMC or UNKNOWN: base is T+1 (next trading day close)
        base_date, base_price = get_next_trading_day_price(symbol, earnings_date)

    if not base_date or base_price is None:
        return result

    result["base_date"] = base_date
    result["base_price"] = base_price

    # Get T+30 trading day price (30 trading days after base_date)
    t30_date, t30_price = get_t_plus_n_trading_day_price(symbol, base_date, 30)

    if not t30_date or t30_price is None:
        return result

    result["t30_date"] = t30_date
    result["t30_price"] = t30_price

    # Calculate percentage change
    if base_price and base_price != 0:
        result["t30_change"] = round((t30_price - base_price) / base_price * 100, 2)

    return result


def fetch_top_gainers(limit: int = TOP_N) -> List[Dict]:
    """Fetch top T+30 gainers from AWS DB (using pre-computed values for ranking)."""
    with get_cursor() as cur:
        if cur is None:
            print("ERROR: Cannot connect to AWS DB", flush=True)
            return []

        cur.execute("""
            SELECT pa.symbol, et.year, et.quarter, et.transcript_date::date as earnings_date,
                   pa.pct_change_t_plus_30
            FROM price_analysis pa
            JOIN earnings_transcripts et ON pa.transcript_id = et.id
            WHERE pa.pct_change_t_plus_30 IS NOT NULL
              AND et.transcript_date >= %s
              AND et.transcript_date < %s
            ORDER BY pa.pct_change_t_plus_30 DESC
            LIMIT %s
        """, (DATE_RANGE_START, DATE_RANGE_END, limit))

        results = []
        for row in cur.fetchall():
            results.append({
                "symbol": row["symbol"],
                "year": row["year"],
                "quarter": row["quarter"],
                "earnings_date": str(row["earnings_date"]),
                "db_t30_change": float(row["pct_change_t_plus_30"]),  # Original DB value for reference
                "category": "GAINER",
                "expected_pred": "UP",
            })
        return results


def fetch_top_losers(limit: int = TOP_N) -> List[Dict]:
    """Fetch top T+30 losers from AWS DB (using pre-computed values for ranking)."""
    with get_cursor() as cur:
        if cur is None:
            print("ERROR: Cannot connect to AWS DB", flush=True)
            return []

        cur.execute("""
            SELECT pa.symbol, et.year, et.quarter, et.transcript_date::date as earnings_date,
                   pa.pct_change_t_plus_30
            FROM price_analysis pa
            JOIN earnings_transcripts et ON pa.transcript_id = et.id
            WHERE pa.pct_change_t_plus_30 IS NOT NULL
              AND et.transcript_date >= %s
              AND et.transcript_date < %s
            ORDER BY pa.pct_change_t_plus_30 ASC
            LIMIT %s
        """, (DATE_RANGE_START, DATE_RANGE_END, limit))

        results = []
        for row in cur.fetchall():
            results.append({
                "symbol": row["symbol"],
                "year": row["year"],
                "quarter": row["quarter"],
                "earnings_date": str(row["earnings_date"]),
                "db_t30_change": float(row["pct_change_t_plus_30"]),  # Original DB value for reference
                "category": "LOSER",
                "expected_pred": "DOWN",
            })
        return results


def determine_hit(predicted: str, actual_change: Optional[float]) -> str:
    """Determine if prediction was correct (hit/miss)."""
    if actual_change is None:
        return "N/A"
    if predicted in ("N/A", "UNKNOWN", None, ""):
        return "N/A"
    pred_upper = str(predicted).upper()
    if pred_upper == "UP":
        return "HIT" if actual_change > 0 else "MISS"
    elif pred_upper == "DOWN":
        return "HIT" if actual_change < 0 else "MISS"
    elif pred_upper == "NEUTRAL":
        return "SKIP"
    return "N/A"


# Global tracking
progress_lock = asyncio.Lock()
results_list: List[Dict] = []
start_time: float = 0
last_report_time: float = 0


def print_progress_report(results: List[Dict], elapsed_seconds: float, is_final: bool = False):
    """Print progress report in table format."""
    report_type = "FINAL RESULTS" if is_final else "PROGRESS REPORT"

    print(f"\n{'='*140}", flush=True)
    print(f"{report_type} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Elapsed: {elapsed_seconds/60:.1f} minutes", flush=True)
    print(f"{'='*140}", flush=True)

    # Table header
    print(f"\n{'Symbol':<8} {'Year':<6} {'Q':<3} {'Date':<12} {'Session':<8} {'Pred':<8} {'Conf':<6} {'T+30%':<10} {'Result':<8}", flush=True)
    print("-" * 100, flush=True)

    # Sort by rank for display
    sorted_results = sorted(results, key=lambda x: x.get("rank", 999))

    for r in sorted_results:
        symbol = r.get("symbol", "N/A")
        year = r.get("year", "N/A")
        quarter = f"Q{r.get('quarter', '?')}"
        date = r.get("earnings_date", "N/A")[:10]
        session = r.get("session", "N/A")
        predicted = r.get("predicted", "N/A")
        confidence = r.get("confidence", "N/A")
        if confidence and confidence != "N/A":
            confidence = f"{confidence}"
        t30 = r.get("t30_change")
        t30_str = f"{t30:+.2f}%" if t30 is not None else "N/A"
        result = r.get("hit_result", "N/A")

        print(f"{symbol:<8} {year:<6} {quarter:<3} {date:<12} {session:<8} {predicted:<8} {confidence:<6} {t30_str:<10} {result:<8}", flush=True)

    # Summary stats
    success = [r for r in results if r.get("status") == "success"]
    hits = [r for r in success if r.get("hit_result") == "HIT"]
    misses = [r for r in success if r.get("hit_result") == "MISS"]
    skips = [r for r in success if r.get("hit_result") == "SKIP"]

    print(f"\n{'='*100}", flush=True)
    print(f"Processed: {len(results)}/{TOP_N * 2}", flush=True)
    print(f"Success: {len(success)}, Errors: {len(results) - len(success)}", flush=True)

    # Session breakdown
    bmo_count = len([r for r in results if r.get("session") == "BMO"])
    amc_count = len([r for r in results if r.get("session") == "AMC"])
    unknown_count = len([r for r in results if r.get("session") == "UNKNOWN"])
    print(f"Sessions: BMO={bmo_count}, AMC={amc_count}, UNKNOWN={unknown_count}", flush=True)

    if success:
        up_pred = len([r for r in success if r.get("predicted") == "UP"])
        down_pred = len([r for r in success if r.get("predicted") == "DOWN"])
        neutral_pred = len([r for r in success if r.get("predicted") == "NEUTRAL"])
        print(f"Predictions: UP={up_pred}, DOWN={down_pred}, NEUTRAL={neutral_pred}", flush=True)

        print(f"Results: HIT={len(hits)}, MISS={len(misses)}, SKIP={len(skips)}", flush=True)

        trades = len(hits) + len(misses)
        if trades > 0:
            accuracy = len(hits) / trades * 100
            print(f"Accuracy (excl. NEUTRAL): {accuracy:.1f}% ({len(hits)}/{trades})", flush=True)

    # Category breakdown
    gainers = [r for r in success if r.get("category") == "GAINER"]
    losers = [r for r in success if r.get("category") == "LOSER"]

    if gainers:
        gainer_hits = len([r for r in gainers if r.get("hit_result") == "HIT"])
        gainer_trades = len([r for r in gainers if r.get("hit_result") in ("HIT", "MISS")])
        if gainer_trades > 0:
            print(f"GAINER Accuracy: {gainer_hits/gainer_trades*100:.1f}% ({gainer_hits}/{gainer_trades})", flush=True)

    if losers:
        loser_hits = len([r for r in losers if r.get("hit_result") == "HIT"])
        loser_trades = len([r for r in losers if r.get("hit_result") in ("HIT", "MISS")])
        if loser_trades > 0:
            print(f"LOSER Accuracy: {loser_hits/loser_trades*100:.1f}% ({loser_hits}/{loser_trades})", flush=True)

    print(f"{'='*140}\n", flush=True)


async def report_progress_periodically(total: int):
    """Background task to print progress reports periodically."""
    global last_report_time
    while True:
        await asyncio.sleep(REPORT_INTERVAL_SECONDS)
        async with progress_lock:
            if len(results_list) < total:
                elapsed = time.time() - start_time
                print_progress_report(results_list, elapsed, is_final=False)
                last_report_time = time.time()
            else:
                break


async def analyze_single(item: Dict, idx: int, total: int, semaphore: asyncio.Semaphore):
    """Analyze a single ticker."""
    async with semaphore:
        symbol = item["symbol"]
        year = item["year"]
        quarter = item["quarter"]
        earnings_date = item["earnings_date"]
        db_t30_change = item["db_t30_change"]
        category = item["category"]
        expected_pred = item["expected_pred"]

        print(f"[{idx}/{total}] {symbol} {year}-Q{quarter} ({category})...", flush=True)

        # Calculate T+30 based on BMO/AMC
        t30_calc = calculate_t30_change(symbol, year, quarter, earnings_date)

        result_row = {
            "rank": idx,
            "symbol": symbol,
            "year": year,
            "quarter": quarter,
            "earnings_date": earnings_date,
            "category": category,
            "expected_pred": expected_pred,
            "session": t30_calc["session"],
            "base_date": t30_calc["base_date"],
            "base_price": t30_calc["base_price"],
            "t30_date": t30_calc["t30_date"],
            "t30_price": t30_calc["t30_price"],
            "t30_change": t30_calc["t30_change"],
            "db_t30_change": db_t30_change,  # Keep original DB value for comparison
            "predicted": "N/A",
            "confidence": None,
            "hit_result": "N/A",
            "status": "pending",
            "error": None,
            "agent_notes": None,
            "full_response": None,
        }

        try:
            start = time.time()
            analysis_result = await analyze_earnings_async(
                symbol=symbol,
                year=year,
                quarter=quarter,
                skip_cache=True,
            )
            elapsed = time.time() - start

            # Check if analysis succeeded: agentic_result exists and has prediction
            agentic_result = analysis_result.get("agentic_result", {}) if analysis_result else {}
            if analysis_result and agentic_result.get("prediction"):

                result_row["predicted"] = agentic_result.get("prediction", "N/A")
                result_row["confidence"] = agentic_result.get("confidence")
                result_row["hit_result"] = determine_hit(result_row["predicted"], t30_calc["t30_change"])
                result_row["status"] = "success"
                result_row["agent_notes"] = agentic_result.get("notes")
                result_row["full_response"] = json.dumps(agentic_result, default=str)

                t30_str = f"{t30_calc['t30_change']:+.2f}%" if t30_calc['t30_change'] is not None else "N/A"
                print(f"  {symbol}: {result_row['predicted']} (conf={result_row['confidence']}) {t30_calc['session']} T+30={t30_str} -> {result_row['hit_result']} [{elapsed:.0f}s]", flush=True)
            else:
                error_msg = analysis_result.get("error", "Unknown") if analysis_result else "No result"
                result_row["status"] = "error"
                result_row["error"] = error_msg
                print(f"  {symbol}: ERROR - {error_msg} [{elapsed:.0f}s]", flush=True)

        except Exception as e:
            result_row["status"] = "error"
            result_row["error"] = str(e)
            print(f"  {symbol}: EXCEPTION - {e}", flush=True)

        async with progress_lock:
            results_list.append(result_row)

        return result_row


def save_results_to_csv(results: List[Dict], output_file: str):
    """Save results to CSV file."""
    if not results:
        print("No results to save", flush=True)
        return

    # Define column order
    columns = [
        "rank", "symbol", "year", "quarter", "earnings_date", "category",
        "session", "base_date", "base_price", "t30_date", "t30_price",
        "expected_pred", "predicted", "confidence", "t30_change", "db_t30_change",
        "hit_result", "status", "error", "agent_notes", "full_response"
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()

        # Sort by rank
        sorted_results = sorted(results, key=lambda x: x.get("rank", 999))

        for row in sorted_results:
            # Truncate long fields
            write_row = row.copy()
            for field in ["agent_notes", "full_response"]:
                if write_row.get(field) and len(str(write_row[field])) > 10000:
                    write_row[field] = str(write_row[field])[:10000] + "...[truncated]"
            writer.writerow(write_row)

    print(f"\nCSV saved to: {output_file}", flush=True)


async def main():
    global start_time, last_report_time, results_list

    print("=" * 140, flush=True)
    print(f"BATCH TEST - Top {TOP_N} Gainers + Top {TOP_N} Losers from AWS DB", flush=True)
    print(f"Date Range: {DATE_RANGE_START} to {DATE_RANGE_END}", flush=True)
    print(f"Concurrency: {CONCURRENCY}", flush=True)
    print(f"Report Interval: {REPORT_INTERVAL_SECONDS}s", flush=True)
    print(f"T+30 Calculation: BMO -> T+30 vs T, AMC -> T+30 vs T+1", flush=True)
    print(f"Using default prompts", flush=True)
    print(f"Started: {datetime.now().isoformat()}", flush=True)
    print("=" * 140, flush=True)

    # Fetch test data from AWS DB
    print("\nFetching test data from AWS DB...", flush=True)

    gainers = fetch_top_gainers(TOP_N)
    print(f"  Fetched {len(gainers)} top gainers", flush=True)

    losers = fetch_top_losers(TOP_N)
    print(f"  Fetched {len(losers)} top losers", flush=True)

    if not gainers and not losers:
        print("ERROR: No test data fetched from AWS DB", flush=True)
        return []

    # Combine test list
    test_list = gainers + losers
    total = len(test_list)
    print(f"\nTotal test cases: {total}", flush=True)

    # Display test data summary
    print("\n--- GAINERS (top 5) ---", flush=True)
    for i, g in enumerate(gainers[:5], 1):
        print(f"  {i}. {g['symbol']} {g['year']}-Q{g['quarter']} ({g['earnings_date']}) DB_T+30={g['db_t30_change']:+.2f}%", flush=True)

    print("\n--- LOSERS (top 5) ---", flush=True)
    for i, l in enumerate(losers[:5], 1):
        print(f"  {i}. {l['symbol']} {l['year']}-Q{l['quarter']} ({l['earnings_date']}) DB_T+30={l['db_t30_change']:+.2f}%", flush=True)

    print("\n" + "=" * 140, flush=True)
    print("Starting analysis...\n", flush=True)

    start_time = time.time()
    last_report_time = start_time
    results_list = []

    semaphore = asyncio.Semaphore(CONCURRENCY)
    reporter_task = asyncio.create_task(report_progress_periodically(total))

    tasks = [
        analyze_single(item, idx, total, semaphore)
        for idx, item in enumerate(test_list, 1)
    ]

    await asyncio.gather(*tasks)

    reporter_task.cancel()
    try:
        await reporter_task
    except asyncio.CancelledError:
        pass

    elapsed = time.time() - start_time

    # Final report
    print_progress_report(results_list, elapsed, is_final=True)

    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(__file__).parent / f"batch_50_results_{timestamp}.csv"
    save_results_to_csv(results_list, str(output_file))

    # Also save as JSON for full data
    json_file = Path(__file__).parent / f"batch_50_results_{timestamp}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2, default=str)
    print(f"JSON saved to: {json_file}", flush=True)

    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"Average per ticker: {elapsed/total:.1f}s", flush=True)

    return results_list


if __name__ == "__main__":
    asyncio.run(main())
