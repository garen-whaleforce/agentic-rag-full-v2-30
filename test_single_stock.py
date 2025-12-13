#!/usr/bin/env python3
"""Test single stock analysis."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

print('=== Environment Verified ===')
print(f'API Version: {os.getenv("AZURE_OPENAI_API_VERSION")}')
print(f'Deployment GPT5: {os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT5")}')
print(f'Deployment GPT4O: {os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4O")}')
print()

# Test analyze
print('=== Testing Single Stock Analysis (SMCI 2024-Q2) ===')
from analysis_engine import analyze_earnings_async


async def main():
    start = time.time()
    result = await analyze_earnings_async('SMCI', 2024, 2, skip_cache=True)
    elapsed = time.time() - start

    print(f'\n=== Result (in {elapsed:.1f}s) ===')
    if result:
        # Print full result structure (keys only)
        print(f'Top-level keys: {list(result.keys())}')

        # Print agentic_result structure
        agentic = result.get('agentic_result', {})
        print(f'\nagentic_result keys: {list(agentic.keys()) if isinstance(agentic, dict) else "NOT A DICT"}')

        if isinstance(agentic, dict):
            print(f'  prediction: {agentic.get("prediction")}')
            print(f'  confidence: {agentic.get("confidence")}')
            summary = agentic.get('summary', '')
            print(f'  summary length: {len(summary) if summary else 0} chars')
            if summary:
                print(f'  summary preview: {str(summary)[:300]}...')

            # Print reasons
            reasons = agentic.get('reasons', [])
            print(f'  reasons: {reasons[:3] if reasons else []}')

        # Backtest
        backtest = result.get('backtest', {})
        if backtest:
            print(f'\nbacktest.session: {backtest.get("session")}')
            print(f'backtest.change_pct: {backtest.get("change_pct")}')
            print(f'backtest.earnings_date: {backtest.get("earnings_date")}')

        # Context
        ctx = result.get('context', {})
        if ctx:
            print(f'\ncontext keys: {list(ctx.keys())[:10]}...')

        print('\n=== Field Completeness Check ===')
        # Check actual fields from agentic_result
        if isinstance(agentic, dict):
            required = ['prediction', 'confidence', 'summary', 'reasons']
            for field in required:
                val = agentic.get(field)
                status = 'OK' if val else 'MISSING/EMPTY'
                print(f'agentic_result.{field}: {status}')
    else:
        print('ERROR: No result returned!')


if __name__ == "__main__":
    asyncio.run(main())
