#!/usr/bin/env python3
"""
Retry failed samples from garen1212v4 batch test.
Loads the previous results, extracts failed samples, and re-runs them.
"""

import json
import time
import csv
import re
import asyncio
import aiohttp
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Configuration
API_URL = "http://localhost:8001/api/analyze"
PREVIOUS_RESULTS_FILE = Path(__file__).parent / "batch_2000_garen1212v4_20251213_234616.json"
OUTPUT_DIR = Path(__file__).parent
TIMEOUT = 300  # 5 minutes per request
PROGRESS_INTERVAL = 10  # Report progress every N samples
CONCURRENCY = 10  # Number of parallel requests


def extract_direction_score(summary: str) -> int:
    """Extract Direction score from summary text."""
    match = re.search(r'Direction\s*[:\s]+(\d+)', summary)
    if match:
        return int(match.group(1))
    return -1


def determine_hit_result(predicted: str, t30_actual: float) -> str:
    """Determine if prediction was correct."""
    if predicted == "NEUTRAL" or t30_actual is None:
        return "SKIP"

    if predicted == "UP":
        return "HIT" if t30_actual > 0 else "MISS"
    elif predicted == "DOWN":
        return "HIT" if t30_actual < 0 else "MISS"

    return "SKIP"


async def analyze_single(
    session: aiohttp.ClientSession,
    sample: Dict,
    idx: int,
    total: int,
    semaphore: asyncio.Semaphore,
    stats: Dict,
    results: List,
    lock: asyncio.Lock
) -> Dict:
    """Analyze a single sample with semaphore for concurrency control."""
    symbol = sample['symbol']
    year = sample['year']
    quarter = sample['quarter']
    t30_actual = sample.get('t30_actual')

    result = {
        'rank': sample.get('rank', idx + 1),
        'symbol': symbol,
        'year': year,
        'quarter': quarter,
        'earnings_date': sample.get('earnings_date'),
        'session': sample.get('session'),
        'category': sample.get('category'),
        't30_actual': t30_actual,
        'v2_predicted': sample.get('v2_predicted'),
        'v2_hit_result': sample.get('v2_hit_result'),
    }

    async with semaphore:
        try:
            req_start = time.time()
            async with session.post(
                API_URL,
                json={
                    'symbol': symbol,
                    'year': year,
                    'quarter': quarter,
                    'refresh': True
                },
                timeout=aiohttp.ClientTimeout(total=TIMEOUT)
            ) as resp:
                elapsed = time.time() - req_start

                if resp.status == 200:
                    data = await resp.json()
                    agentic = data.get('agentic_result', {})

                    predicted = agentic.get('prediction', 'UNKNOWN')
                    confidence = agentic.get('confidence', 0)
                    summary = agentic.get('summary', '')
                    raw_output = agentic.get('raw_output', '') or summary

                    direction_score = extract_direction_score(summary)
                    hit_result = determine_hit_result(predicted, t30_actual)

                    result.update({
                        'status': 'success',
                        'predicted': predicted,
                        'confidence': confidence,
                        'direction_score': direction_score,
                        'hit_result': hit_result,
                        'elapsed_seconds': round(elapsed, 1),
                        'main_summary': summary,
                        'raw_output': raw_output,
                        'notes_financials': agentic.get('notes_financials', ''),
                        'notes_past': agentic.get('notes_past', ''),
                        'notes_peers': agentic.get('notes_peers', ''),
                        'error': ''
                    })

                    async with lock:
                        stats['success'] += 1
                        stats[predicted.lower()] = stats.get(predicted.lower(), 0) + 1
                        stats[hit_result.lower()] += 1

                else:
                    text = await resp.text()
                    result.update({
                        'status': 'error',
                        'predicted': '',
                        'confidence': 0,
                        'direction_score': -1,
                        'hit_result': 'ERROR',
                        'elapsed_seconds': round(elapsed, 1),
                        'main_summary': '',
                        'raw_output': '',
                        'error': f"HTTP {resp.status}: {text[:200]}"
                    })
                    async with lock:
                        stats['error'] += 1

        except asyncio.TimeoutError:
            result.update({
                'status': 'timeout',
                'predicted': '',
                'confidence': 0,
                'direction_score': -1,
                'hit_result': 'ERROR',
                'elapsed_seconds': TIMEOUT,
                'main_summary': '',
                'raw_output': '',
                'error': 'Request timeout'
            })
            async with lock:
                stats['timeout'] += 1

        except Exception as e:
            result.update({
                'status': 'exception',
                'predicted': '',
                'confidence': 0,
                'direction_score': -1,
                'hit_result': 'ERROR',
                'elapsed_seconds': 0,
                'main_summary': '',
                'raw_output': '',
                'error': str(e)[:200]
            })
            async with lock:
                stats['error'] += 1

    # Per-sample output with t30 and hit result
    status_char = "✓" if result.get('status') == 'success' else "✗"
    pred = result.get('predicted', 'N/A')
    hit = result.get('hit_result', 'N/A')
    t30 = t30_actual
    t30_str = f"{t30:+.1f}%" if t30 is not None else "N/A"
    elapsed_s = result.get('elapsed_seconds', 0)
    conf = result.get('confidence', 0)

    print(f"  [{idx+1}/{total}] {status_char} {symbol} {year}Q{quarter}: {pred}({conf}) -> {hit} | t30={t30_str} ({elapsed_s:.0f}s)", flush=True)

    async with lock:
        results.append(result)

    return result


async def run_retry():
    import sys
    sys.stdout.reconfigure(line_buffering=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = OUTPUT_DIR / f"batch_2000_garen1212v4_retry_{timestamp}.json"
    output_csv = OUTPUT_DIR / f"batch_2000_garen1212v4_retry_{timestamp}.csv"
    merged_json = OUTPUT_DIR / f"batch_2000_garen1212v4_merged_{timestamp}.json"
    merged_csv = OUTPUT_DIR / f"batch_2000_garen1212v4_merged_{timestamp}.csv"

    # Load previous results
    with open(PREVIOUS_RESULTS_FILE, 'r') as f:
        previous_results = json.load(f)

    # Separate successful and failed samples
    successful_results = [r for r in previous_results if r.get('status') == 'success']
    failed_samples = [r for r in previous_results if r.get('status') != 'success']

    total = len(failed_samples)
    print(f"=" * 70)
    print(f"GAREN1212V4 RETRY - Failed Samples")
    print(f"=" * 70)
    print(f"Previous successful: {len(successful_results)}")
    print(f"Failed to retry: {total}")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Output JSON: {output_json.name}")
    print(f"Output CSV: {output_csv.name}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=" * 70)
    print(f"Format: [n/total] status SYMBOL YYYYQn: PRED(conf) -> HIT/MISS | t30=+X.X% (time)")
    print(f"=" * 70)

    results = []
    stats = {
        'success': 0,
        'error': 0,
        'timeout': 0,
        'up': 0,
        'down': 0,
        'neutral': 0,
        'hit': 0,
        'miss': 0,
        'skip': 0
    }

    start_time = time.time()
    semaphore = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()

    connector = aiohttp.TCPConnector(limit=CONCURRENCY * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        async def process_with_progress():
            tasks = []
            for idx, sample in enumerate(failed_samples):
                task = asyncio.create_task(
                    analyze_single(session, sample, idx, total, semaphore, stats, results, lock)
                )
                tasks.append(task)

            completed_count = 0
            last_report = 0

            for coro in asyncio.as_completed(tasks):
                await coro
                completed_count += 1

                if completed_count - last_report >= PROGRESS_INTERVAL or completed_count == total:
                    last_report = completed_count
                    elapsed_total = time.time() - start_time
                    avg_time = elapsed_total / completed_count if completed_count > 0 else 0
                    eta = avg_time * (total - completed_count)

                    valid = stats['hit'] + stats['miss']
                    accuracy = stats['hit'] / valid * 100 if valid > 0 else 0

                    print(f"\n{'='*70}")
                    print(f"[Progress {completed_count}/{total}] ({completed_count/total*100:.1f}%)")
                    print(f"  Success: {stats['success']}, Error: {stats['error']}, Timeout: {stats['timeout']}")
                    print(f"  UP: {stats['up']}, DOWN: {stats['down']}, NEUTRAL: {stats['neutral']}")
                    print(f"  HIT: {stats['hit']}, MISS: {stats['miss']}, SKIP: {stats['skip']}")
                    print(f"  Retry Accuracy: {stats['hit']}/{valid} = {accuracy:.1f}%")
                    print(f"  Elapsed: {elapsed_total/60:.1f}min, ETA: {eta/60:.1f}min")
                    print(f"{'='*70}\n")

                    # Save intermediate results
                    sorted_results = sorted(results, key=lambda x: x['rank'])
                    with open(output_json, 'w') as f:
                        json.dump(sorted_results, f, indent=2, ensure_ascii=False)

        await process_with_progress()

    # Retry summary
    total_time = time.time() - start_time
    valid = stats['hit'] + stats['miss']
    accuracy = stats['hit'] / valid * 100 if valid > 0 else 0

    print(f"\n{'='*70}")
    print(f"RETRY RESULTS - garen1212v4")
    print(f"{'='*70}")
    print(f"Retried samples: {total}")
    print(f"Success: {stats['success']} ({stats['success']/total*100:.1f}%)")
    print(f"Still failing: {stats['error'] + stats['timeout']}")
    print(f"Retry Accuracy: {stats['hit']}/{valid} = {accuracy:.1f}%")
    print(f"Total time: {total_time/60:.1f} minutes")

    # Save retry results
    sorted_results = sorted(results, key=lambda x: x['rank'])
    with open(output_json, 'w') as f:
        json.dump(sorted_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved retry JSON: {output_json}")

    # Merge with successful results
    retry_success = [r for r in sorted_results if r.get('status') == 'success']
    merged_results = successful_results + retry_success
    merged_results = sorted(merged_results, key=lambda x: x['rank'])

    # Calculate merged stats
    merged_hits = len([r for r in merged_results if r.get('hit_result') == 'HIT'])
    merged_misses = len([r for r in merged_results if r.get('hit_result') == 'MISS'])
    merged_valid = merged_hits + merged_misses
    merged_accuracy = merged_hits / merged_valid * 100 if merged_valid > 0 else 0

    print(f"\n{'='*70}")
    print(f"MERGED RESULTS (Previous + Retry)")
    print(f"{'='*70}")
    print(f"Total successful: {len(merged_results)} / 2000")
    print(f"HIT: {merged_hits}, MISS: {merged_misses}")
    print(f"Overall Accuracy: {merged_hits}/{merged_valid} = {merged_accuracy:.1f}%")

    # Save merged results
    with open(merged_json, 'w') as f:
        json.dump(merged_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved merged JSON: {merged_json}")

    # Save merged CSV
    csv_columns = [
        'rank', 'symbol', 'year', 'quarter', 'earnings_date', 'session', 'category',
        't30_actual', 'v2_predicted', 'v2_hit_result',
        'status', 'predicted', 'confidence', 'direction_score', 'hit_result',
        'elapsed_seconds', 'main_summary', 'raw_output',
        'notes_financials', 'notes_past', 'notes_peers', 'error'
    ]

    with open(merged_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(merged_results)
    print(f"Saved merged CSV: {merged_csv}")

    return merged_results


if __name__ == "__main__":
    asyncio.run(run_retry())
