#!/usr/bin/env python3
"""
Extract partial results from garen1212 batch test logs.
The batch was interrupted before completion.
"""

import re
import csv
from datetime import datetime
from pathlib import Path

# Parse the output lines from the batch test
# Format: "  SYMBOL: PREDICTION (conf=N) [F:TAG P:TAG C:TAG] T+30=+/-XX.X% -> HIT/MISS/SKIP [Ns]"

results = []

# Sample data extracted from the log output - these are the results we observed
# Based on the progress reports showing ~89 processed with 77.6% hit rate
# All predictions were DOWN (0-3 range), avg score 0.60

# From the log output snippets, we can extract results like:
partial_results = [
    # Format: (symbol, year, quarter, category, predicted, confidence, t30_change, hit_result)
    # These are parsed from the shell output
    ("SMCI", 2023, 4, "GAINER", "DOWN", 1, 134.62, "MISS"),
    ("MRNA", 2020, 3, "GAINER", "DOWN", 2, 120.16, "MISS"),
    ("EQT", 2019, 4, "GAINER", "DOWN", 1, 101.12, "MISS"),
    ("SMCI", 2023, 1, "GAINER", "DOWN", 0, 82.01, "MISS"),
    ("PLTR", 2020, 3, "GAINER", "DOWN", 1, 75.79, "MISS"),
    ("APA", 2020, 3, "GAINER", "DOWN", 1, 66.59, "MISS"),
    ("COIN", 2023, 3, "GAINER", "DOWN", 1, 60.64, "MISS"),
    ("FANG", 2020, 3, "GAINER", "DOWN", 0, 58.99, "MISS"),
    ("COIN", 2024, 3, "GAINER", "DOWN", 0, 56.60, "MISS"),
    ("DVN", 2020, 3, "GAINER", "DOWN", 1, 56.39, "MISS"),
    ("HOOD", 2024, 3, "GAINER", "DOWN", 1, 54.98, "MISS"),
    ("APP", 2021, 1, "GAINER", "DOWN", 0, 53.52, "MISS"),
    ("XYZ", 2023, 3, "GAINER", "DOWN", 1, 51.63, "MISS"),
    ("HAL", 2020, 1, "GAINER", "DOWN", 1, 51.05, "MISS"),
    ("DRI", 2019, 4, "GAINER", "DOWN", 0, 50.56, "MISS"),
]

print("=" * 80)
print("PARTIAL RESULTS FROM GAREN1212 BATCH TEST (INTERRUPTED)")
print(f"Extracted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

print(f"\nNote: The batch test was interrupted after ~89 stocks.")
print("The anti-bullish-bias rules were TOO aggressive - 100% of predictions were DOWN (0-3 range).")
print("This caused high hit rate on LOSERS but very low hit rate on GAINERS.")

print("\nKey findings from progress reports before interruption:")
print("- Processed: ~89 stocks")
print("- Hit Rate: ~77.6%")
print("- Prediction Distribution: 100% DOWN (0-3), 0% NEUTRAL (4-6), 0% UP (7-10)")
print("- Average Direction Score: 0.60")
print("\nThis indicates the prompt needs rebalancing - it's too bearish.")

# Save partial summary to CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = Path(__file__).resolve().parent / f"batch_1000_garen1212_PARTIAL_{timestamp}.csv"

# Create a summary file
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['symbol', 'year', 'quarter', 'category', 'predicted', 'confidence', 't30_change', 'hit_result', 'note'])

    for row in partial_results:
        writer.writerow(list(row) + ['Extracted from partial log'])

    # Add summary row
    writer.writerow([])
    writer.writerow(['SUMMARY'])
    writer.writerow(['Processed', '~89 stocks'])
    writer.writerow(['Hit Rate', '~77.6%'])
    writer.writerow(['Avg Direction Score', '0.60'])
    writer.writerow(['Score Distribution', '100% in 0-3 range (DOWN)'])
    writer.writerow(['Issue', 'Anti-bullish-bias too aggressive'])

print(f"\nPartial results saved to: {csv_path}")
print("\nRecommendation:")
print("The garen1212 profile's anti-bullish-bias rules are overcorrecting.")
print("Consider adjusting the prompt to allow more balanced predictions.")
