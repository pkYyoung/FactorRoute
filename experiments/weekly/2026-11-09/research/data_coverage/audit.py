"""Audit the longest contiguous recent range with usable free daily data."""

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import run as pit


FIRST = "2021-01-01"
LAST = "2026-10-30"
FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
MISSING_CLOSE = {"unresolved_identity_or_source", "no_signal_close"}
FACTORS = {"eligible", "liquidity_filter"}
VERIFIED_LISTINGS = {
    ("CEG", "0001868275"): "2022-02-02",  # Constellation separation announcement.
    ("GEHC", "0001932393"): "2023-01-04",  # GE HealthCare launch announcement.
    ("KVUE", "0001944048"): "2023-05-04",  # Kenvue IPO announcement.
    ("GEV", "0001996810"): "2024-04-02",  # GE Vernova investor FAQ.
}
FISERV_CIK = "0000798354"
FI_FIRST = pd.Timestamp("2023-06-07")
FISV_RETURN = pd.Timestamp("2025-11-11")


def words(name):
    stop = {"a", "and", "the", "inc", "corporation", "corp", "plc", "company", "co", "ltd", "group", "holdings", "class"}
    normalized = re.sub(r"(?<=[a-z])\.(?=[a-z])", "", (name or "").lower())
    return [word for word in re.findall(r"[a-z0-9]+", normalized) if word not in stop]


def same_name(historical, metadata):
    expected = words(historical)
    for field in ("longName", "shortName"):
        observed = words(metadata.get(field))
        if expected and (expected == observed or (observed and expected[0] == observed[0] and len(expected[0]) >= 3)):
            return True
    return False


def verified_downloads(folder, history, market):
    results = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["results"]
    current = market["universe"].set_index("code").issuer_id.to_dict()
    accepted, ledger = {}, []
    for result in results:
        symbol = result["requested"]
        rows = history.loc[history.symbol == symbol]
        reason = result["status"]
        if reason == "downloaded":
            if result["symbol"] != symbol or result["instrumentType"] != "EQUITY" or result["currency"] != "USD":
                reason = "wrong_symbol_instrument_or_currency"
            else:
                matching = rows.loc[rows.name.map(lambda name: same_name(name, result))]
                issuers = set(matching.cik) - {""}
                if len(issuers) != 1 or (symbol in current and current[symbol] not in issuers):
                    reason = "unverified_or_reused_ticker"
                else:
                    issuer = issuers.pop()
                    frame = pd.read_csv(folder / f"{symbol}.csv", index_col="date", parse_dates=True)
                    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
                        raise ValueError(f"Unsorted or duplicate bars: {symbol}")
                    valid = (np.isfinite(frame[FIELDS]).all(axis=1) & (frame[FIELDS[:-1]] > 0).all(axis=1)
                             & (frame.Volume >= 0) & (frame.High >= frame[["Open", "Close", "Low"]].max(axis=1))
                             & (frame.Low <= frame[["Open", "Close", "High"]].min(axis=1)))
                    frame = frame.loc[valid]
                    accepted[(symbol, issuer)] = frame.reindex(market["close"].index)
                    reason = "accepted_for_coverage"
        ledger.append({**result, "decision": reason})
    return accepted, pd.DataFrame(ledger)


def summarize(archive, end):
    weekly = archive.groupby("signal").agg(
        members=("status", "size"), close_rate_before=("close_before", "mean"),
        factor_rate_before=("factor_before", "mean"), close_rate=("has_close", "mean"),
        raw_factor_rate=("factor_ready", "mean"), approx_date_rows=("approximate_date", "sum"))
    weekly["factor_rate"] = archive.loc[~archive.verified_short_listing].groupby("signal").factor_ready.mean()
    opens = archive.loc[archive.liquid].groupby("signal").has_next_open.mean()
    weekly["next_open_rate"] = opens
    weekly.loc[end, "next_open_rate"] = np.nan
    weekly["numeric_gate"] = (weekly.close_rate >= .95) & (weekly.factor_rate >= .95) & (weekly.next_open_rate >= .98)
    weekly.loc[end, "numeric_gate"] = False
    # Oct 30 is a terminal mark, with no following rebalance in the tested period.
    effective = weekly.loc[weekly.index < end]
    failed = effective.index[~effective.numeric_gate]
    start = effective.index[0] if failed.empty else effective.index[effective.index > failed[-1]].min()
    return weekly, start


def build_coverage(root):
    market = pit.load_snapshot(pit.SNAPSHOT, LAST)
    dates = market["close"].index
    (close, opening, volume, dollar), _ = pit.extra_prices(pit.INPUT / "yahoo", dates, market)
    history = pit.history(pit.INPUT / "components_history.csv")
    accepted, ledger = verified_downloads(root / "yahoo", history, market)
    archive = pd.read_csv(pit.INPUT / "results-verified/member_mapping_and_gaps.csv", dtype={"CIK": str},
                          parse_dates=["signal"], low_memory=False)
    archive = archive.loc[(archive.pool == "augmented_prices") & (archive.signal >= FIRST)].copy()
    archive["CIK"] = archive.CIK.fillna("")
    archive["quote_code"] = archive.quote_code.fillna("")
    # The community file has an evergreen FISV row overlapping the historical FI code.
    # Fiserv's SEC filings confirm FI in the interval below and FISV outside it.
    in_fi_period = (archive.signal >= FI_FIRST) & (archive.signal < FISV_RETURN)
    fiserv = archive.CIK == FISERV_CIK
    archive = archive.loc[~(fiserv & (in_fi_period == (archive.historical_symbol == "FISV")))].copy()
    archive.loc[archive.CIK == FISERV_CIK, "quote_code"] = "FISV"
    archive["close_before"] = ~archive.status.isin(MISSING_CLOSE)
    archive["factor_before"] = archive.status.isin(FACTORS)
    archive["sector"] = ""
    for signal, rows in archive.groupby("signal"):
        members = pit.members_on(history, signal).set_index(["symbol", "cik"])
        ids = pd.MultiIndex.from_frame(rows[["historical_symbol", "CIK"]])
        archive.loc[rows.index, "sector"] = members.loc[ids, "sector"].to_numpy()

    weekly_next = {date: dates[i + 1] if i + 1 < len(dates) else pd.NaT
                   for date, i in zip(archive.signal.unique(), dates.get_indexer(archive.signal.unique()), strict=True)}
    prepared = {}
    for key, frame in accepted.items():
        prepared[key] = (frame["Adj Close"], frame.Open * frame["Adj Close"] / frame.Close,
                         frame.Volume, frame.Close * frame.Volume)
    parts = []
    all_prices = {}
    for identity, rows in archive.groupby(["historical_symbol", "CIK", "quote_code"], sort=False):
        symbol, issuer, mapped = identity
        idx = pd.DatetimeIndex(rows.signal)
        listed = VERIFIED_LISTINGS.get((symbol, issuer))
        rows = rows.assign(verified_short_listing=(False if listed is None else
                           idx < dates[dates.searchsorted(listed) + 273]))
        options = []
        if mapped in close:
            options.append(("archived", close[mapped], opening[mapped], volume[mapped], dollar[mapped]))
        if (symbol, issuer) in prepared:
            options.append(("new_yahoo", *prepared[symbol, issuer]))
        if not options:
            rows = rows.assign(has_close=False, factor_ready=False, liquid=False, has_next_open=False,
                               price_source="missing", adv20=np.nan)
            parts.append(rows)
            continue
        # Keep one internally consistent history per mapping. Never stitch different adjustment bases.
        price_source, cl, op, vol, adv_input = max(options, key=lambda v: (
            v[1].notna().rolling(274, min_periods=274).sum().eq(274).reindex(idx).sum(),
            v[1].reindex(idx).notna().sum()))
        adv = adv_input.rolling(20, min_periods=20).mean().reindex(idx)
        has_close = cl.reindex(idx).notna().to_numpy()
        warm = cl.notna().rolling(274, min_periods=274).sum().eq(274).reindex(idx).to_numpy()
        ready = warm & adv.notna().to_numpy()
        liquid = ready & (adv.to_numpy() >= 10_000_000) & (vol.reindex(idx).to_numpy() > 0) & (rows.sector != "").to_numpy()
        following = rows.signal.map(weekly_next)
        has_open = (op.reindex(following).notna() & (vol.reindex(following) > 0)).to_numpy()
        rows = rows.assign(has_close=has_close, factor_ready=ready, liquid=liquid, has_next_open=has_open,
                           price_source=price_source, adv20=adv.to_numpy())
        parts.append(rows)
        all_prices[identity] = (cl, op, vol)

    archive = pd.concat(parts).sort_values(["signal", "historical_symbol", "CIK"])
    return market, archive, all_prices, weekly_next, ledger


def audit(root):
    market, archive, all_prices, weekly_next, ledger = build_coverage(root)
    dates = market["close"].index
    ledger.to_csv(root / "source_decisions.csv", index=False)
    exit_rows = []
    end = pd.Timestamp(LAST)
    weekly, start = summarize(archive, end)
    weekly.to_csv(root / "weekly_coverage.csv")
    archive.loc[~archive.factor_ready | (archive.liquid & ~archive.has_next_open) | archive.approximate_date
                ].to_csv(root / "weekly_gaps.csv", index=False)
    weekly.groupby(weekly.index.year).agg(
        weeks=("members", "size"), close_before_min=("close_rate_before", "min"), close_min=("close_rate", "min"),
        close_median=("close_rate", "median"), factor_min=("factor_rate", "min"), factor_median=("factor_rate", "median"),
        opening_min=("next_open_rate", "min"), numeric_pass_weeks=("numeric_gate", "sum"),
    ).to_csv(root / "annual_coverage.csv")
    spy = market["benchmark"]["Adj Close"].reindex(weekly.index).pct_change(fill_method=None)
    archive["spy_week"] = archive.signal.map(spy)
    archive.assign(spy_down_2pct=archive.spy_week <= -.02).groupby(
        [archive.signal.dt.year, "sector", "spy_down_2pct"], dropna=False
    ).agg(member_weeks=("status", "size"), close_rate=("has_close", "mean"),
          raw_factor_rate=("factor_ready", "mean")).to_csv(root / "sector_stress_coverage.csv")

    groups = list(archive.groupby("signal"))
    for (signal, previous), (next_signal, current) in zip(groups, groups[1:]):
        current_ids = set(zip(current.historical_symbol, current.CIK, strict=True))
        for row in previous.itertuples(index=False):
            if (row.historical_symbol, row.CIK) in current_ids:
                continue
            first, sell = weekly_next[signal], weekly_next[next_signal]
            prices = all_prices.get((row.historical_symbol, row.CIK, row.quote_code))
            hold_marks = sale_open = False
            if prices is not None and pd.notna(sell):
                cl, op, vol = prices
                hold_marks = cl.loc[(dates >= first) & (dates < sell)].notna().all()
                sale_open = bool(pd.notna(op.at[sell]) and vol.at[sell] > 0)
            exit_rows.append({"last_member_signal": signal, "next_signal": next_signal, "scheduled_exit": sell,
                              "symbol": row.historical_symbol, "CIK": row.CIK, "sector": row.sector,
                              "could_rank": row.liquid, "price_source": row.price_source,
                              "complete_hold_marks": hold_marks, "has_sale_open": sale_open,
                              "same_issuer_still_member": row.CIK in set(current.CIK),
                              "approximate_date": row.approximate_date})
    exits = pd.DataFrame(exit_rows)
    exits.to_csv(root / "membership_exit_checks.csv", index=False)
    usable_weeks = int(((weekly.index >= start) & (weekly.index < end)).sum())
    report = {"first_contiguous_numeric_signal": None if pd.isna(start) else str(start.date()),
              "first_contiguous_numeric_open": None if pd.isna(start) else str(weekly_next[start].date()),
              "terminal_close": LAST, "accepted_price_series": int(ledger.decision.eq("accepted_for_coverage").sum()),
              "weeks": usable_weeks,
              "rolling_13_week_starts": max(0, usable_weeks - 12), "disjoint_13_week_windows": usable_weeks // 13}
    (root / "range.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    audit(parser.parse_args().root)
