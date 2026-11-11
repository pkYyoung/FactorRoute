"""Download public Yahoo bars for documented recent dynamic-universe gaps."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

from ..strategy_comparison.run import ROOT

import pandas as pd
import yfinance as yf


PILOT = ROOT / "outputs/weekly/2026-11-09/free-pit-pilot"
FIRST = "2021-01-01"


def fetch(ticker, out):
    stock = yf.Ticker(ticker)
    record = {"requested": ticker}
    try:
        prices = stock.history(start="2019-01-01", end="2026-10-31", auto_adjust=False, actions=True)
        if prices.empty:
            record["status"] = "no_history"
            return record
        meta = stock.get_history_metadata()
        record.update({key: meta.get(key) for key in ("symbol", "longName", "shortName", "exchangeName", "instrumentType", "currency")})
        prices.index = prices.index.tz_localize(None)
        prices.index.name = "date"
        prices.to_csv(out / f"{ticker}.csv", mode="x")
        record.update(status="downloaded", rows=len(prices), first=str(prices.index.min().date()), last=str(prices.index.max().date()))
    except (OSError, ValueError, RuntimeError) as exc:
        record.update(status="error", error=str(exc))
    return record


def main(out):
    archive = pd.read_csv(PILOT / "results-verified/member_mapping_and_gaps.csv", low_memory=False)
    gaps = archive.loc[(archive.pool == "augmented_prices") & (archive.signal >= FIRST)
                       & ~archive.status.isin(["eligible", "liquidity_filter"])]
    symbols = sorted(set(gaps.historical_symbol) | set(gaps.quote_code.dropna()))
    out.mkdir(parents=True)
    yf.set_tz_cache_location(str(out / "cache"))
    with ThreadPoolExecutor(max_workers=5) as workers:
        results = list(workers.map(lambda ticker: fetch(ticker, out), symbols))
    manifest = {"retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "yfinance": yf.__version__, "results": results}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.Series([record["status"] for record in results]).value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    main(parser.parse_args().out)
