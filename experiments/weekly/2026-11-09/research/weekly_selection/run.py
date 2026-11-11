"""Replay the Oct 30 signal and compare the two Nov 6 research portfolios."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..data_coverage.audit import build_coverage, pit
from ..weight_search import grid as grid


OLD = pd.Timestamp("2026-10-30")
SIGNAL = pd.Timestamp("2026-11-06")
WEEK = pd.date_range("2026-11-02", SIGNAL, freq="B")
ROOT = pit.INPUT.parent
OKTA_CIK = "0001660134"
INDEX_CHANGE = "https://press.spglobal.com/2026-10-30-Okta-Set-to-Join-S-P-500-Others-to-Join-S-P-MidCap-400-and-S-P-SmallCap-600"
RECIPES = {"baseline_70_30": grid.BASE, "rank_06": (20, 0, 55, 15, 5, 0, 5, 0, 0)}


def universe():
    market, audit, histories, _, _ = build_coverage(ROOT / "free-data-audit")
    rows = audit.loc[audit.signal == OLD].copy()
    rows["key"] = list(zip(rows.historical_symbol, rows.CIK, rows.quote_code, strict=True))
    rows["code"] = np.where(rows.price_source == "archived", rows.quote_code, rows.historical_symbol)
    return market, rows, histories


def download(out):
    import yfinance as yf

    _, rows, _ = universe()
    codes = sorted(set(rows.loc[rows.price_source != "missing", "code"]) | {"SPY", "OKTA"})
    out.mkdir(parents=True)
    yf.set_tz_cache_location(str(out / "cache"))
    frames = []
    for start in range(0, len(codes), 80):
        part = codes[start:start + 80]
        data = yf.download(part, start=OLD, end=SIGNAL + pd.Timedelta(days=1), auto_adjust=False,
                           actions=True, progress=False, threads=8)
        frames.append(data)
        print(f"Fetched quote batch {start // 80 + 1}: {len(part)} tickers", flush=True)
    bars = pd.concat(frames, axis=1).sort_index(axis=1)
    okta = yf.Ticker("OKTA").history(start="2025-08-01", end=SIGNAL + pd.Timedelta(days=1),
                                    auto_adjust=False, actions=True)
    if okta.empty:
        raise ValueError("No OKTA warm-up history")
    okta.index = okta.index.tz_localize(None)
    okta.to_csv(out / "okta.csv", mode="x")
    bars.to_csv(out / "week_quotes.csv", mode="x")
    pit.dump(out / "download.json", {"requested": codes, "source": "Yahoo Finance via yfinance " + yf.__version__,
                                     "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                                     "week_quotes_sha256": pit.sha(out / "week_quotes.csv"),
                                     "okta_sha256": pit.sha(out / "okta.csv"), "membership_source": INDEX_CHANGE})


def ranked(date, prices, benchmark, pool):
    date_index = prices.index.get_loc(date)
    months = pd.Series(prices.index).groupby(prices.index.to_period("M")).max()
    completed = months.loc[(months <= date) & (months != SIGNAL)].tail(13)
    pool = pool.sort_values(["adv20", "code"], ascending=[False, True]).drop_duplicates("CIK")
    valid = prices.loc[completed, pool.code].notna().all()
    pool = pool.loc[pool.code.isin(valid[valid].index)].sort_values("code")
    codes = pool.code.tolist()
    price_frame = prices[codes].assign(SPY=benchmark)
    i = date_index
    reference = pit.scores(price_frame, benchmark.pct_change(fill_method=None), i, codes)
    sectors = {name: j for j, name in enumerate(sorted(pool.sector.unique()))}
    rows = pd.DataFrame({"symbol_id": np.arange(len(codes)), "code": codes, "sector": pool.sector.to_numpy(),
                         "sector_id": pool.sector.map(sectors).to_numpy(), "HBETA": reference.HBETA.to_numpy(),
                         "z_RSTR": pit.standardize(reference.RSTR).to_numpy(),
                         "z_DASTD": pit.standardize(reference.DASTD).to_numpy()})
    # Include the rest of November's calendar so the factor helper does not mistake Nov 6 for month-end.
    if date == SIGNAL:
        month_end = pd.Timestamp("2026-11-30")
        price_frame = price_frame.reindex(price_frame.index.append(pd.DatetimeIndex([month_end])))
    values = grid.daily.factor_panels(price_frame.index, price_frame.to_numpy(), len(codes), [(i, rows)])[0][3]
    result = rows[["code", "sector"]].copy()
    for name, weights in RECIPES.items():
        scores = values @ weights / 100
        order = np.lexsort((np.asarray(codes), -scores))
        result[name] = np.argsort(order) + 1
    return result


def pick(ranking, recipe, previous=()):
    ordered = ranking.sort_values(recipe)
    priority = ordered.loc[(ordered[recipe] <= 15) & ordered.code.isin(previous)]
    sectors, chosen = Counter(), []
    for row in pd.concat([priority, ordered]).drop_duplicates("code").itertuples():
        if sectors[row.sector] < 3:
            sectors[row.sector] += 1
            chosen.append(row.code)
        if len(chosen) == 10:
            return chosen
    raise ValueError("Industry cap prevents filling the ten-stock portfolio")


def source_dollar(row, market):
    if row.price_source == "archived" and row.code in market["dollar"]:
        return market["dollar"][row.code]
    folder = (pit.INPUT / "yahoo") if row.price_source == "archived" else (ROOT / "free-data-audit/yahoo")
    frame = pd.read_csv(folder / f"{row.code}.csv", index_col=0, parse_dates=True)
    return frame.Close * frame.Volume


def compare(out):
    import json

    manifest = json.loads((out / "download.json").read_text(encoding="utf-8"))
    for name in ("week_quotes", "okta"):
        if pit.sha(out / f"{name}.csv") != manifest[f"{name}_sha256"]:
            raise ValueError(f"Changed quote snapshot: {name}")
    freeze = pd.read_csv(ROOT / "spy-grid-wide/frozen_top20.csv")
    assert tuple(freeze.loc[freeze.training_rank.eq(6), list(grid.NAMES)].iloc[0]) == RECIPES["rank_06"]
    market, members, histories = universe()
    usable = members.loc[members.price_source != "missing"].copy()
    if usable.code.duplicated().any():
        raise ValueError("Multiple active securities map to the same quote")
    archived = pd.read_csv(ROOT / "spy-candidate-pilot/frozen_week_targets.csv")
    last_baseline = archived.loc[archived.recipe == "baseline", "ticker"].tolist()
    retained_archive = pd.read_csv(pit.ROOT / "outputs/weekly/2026-11-09/capped-buffer-week-review/selected.csv")
    assert set(last_baseline) == set(retained_archive.code)
    close = pd.DataFrame({row.code: histories[row.key][0] for row in usable.itertuples()})
    old_rank = ranked(OLD, close, market["benchmark"]["Adj Close"], usable.loc[usable.liquid])
    if set(pick(old_rank, "baseline_70_30")) != set(last_baseline):
        raise ValueError("Rebuilt baseline disagrees with the archived Oct 30 selection")
    old = {"baseline_70_30": last_baseline, "rank_06": pick(old_rank, "rank_06")}

    bars = pd.read_csv(out / "week_quotes.csv", header=[0, 1], index_col=0, parse_dates=True)
    if not bars.index.equals(WEEK.insert(0, OLD)):
        raise ValueError("Quote snapshot does not cover all sessions")
    adjusted = bars["Adj Close"]
    day_open = bars.Open * adjusted / bars.Close
    details, performance, contributions = [], [], []
    for name, codes in {**old, "SPY": ["SPY"]}.items():
        if not (np.isfinite(adjusted.loc[WEEK, codes]).all().all() and (day_open.loc[WEEK[0], codes] > 0).all()
                and (bars.Volume.loc[WEEK, codes] > 0).all().all()):
            raise ValueError(f"Incomplete held prices for {name}")
        relatives = adjusted.loc[WEEK, codes] / day_open.loc[WEEK[0], codes]
        for day, value in (relatives.mean(axis=1) / 1.001).items():
            details.append((day, name, value))
        for code in codes:
            contributions.append((name, code, relatives[code].iloc[-1] - 1, (relatives[code].iloc[-1] - 1) / len(codes)))
        for fee_bps in (10, 15):
            nav = relatives.mean(axis=1) / (1 + fee_bps / 10_000)
            path = np.r_[1., nav]
            performance.append((name, fee_bps, nav.iloc[-1] - 1, (1 - path / np.maximum.accumulate(path)).max()))
    pd.DataFrame(details, columns=["date", "recipe", "nav_at_10bps"]).to_csv(out / "nov02_06_nav.csv", index=False)
    pd.DataFrame(performance, columns=["recipe", "fee_bps", "net_return", "max_drawdown"]).to_csv(out / "nov02_06_results.csv", index=False)
    pd.DataFrame(contributions, columns=["recipe", "ticker", "gross_return", "gross_contribution"]).to_csv(out / "nov02_06_contributions.csv", index=False)

    # Roll the verified Oct 30 membership to Nov 6 using the official Nov 4 replacement.
    current = members.loc[members.historical_symbol.ne("KVUE")].copy()
    assert not current.historical_symbol.eq("OKTA").any()
    new_closes, new_dollars = {}, {}
    for row in usable.loc[usable.historical_symbol.ne("KVUE")].itertuples():
        code = row.code
        anchor = adjusted.at[OLD, code]
        if not (anchor > 0 and close.at[OLD, code] > 0):
            continue
        recent = adjusted.loc[WEEK, code].where((adjusted.loc[WEEK, code] > 0) & (bars.Volume.loc[WEEK, code] > 0))
        new_closes[code] = pd.concat([close[code], recent * (close.at[OLD, code] / anchor)])
        new_dollars[code] = pd.concat([source_dollar(row, market).loc[:OLD],
                                      (bars.Close[code] * bars.Volume[code]).loc[WEEK]])
    okta = pd.read_csv(out / "okta.csv", index_col=0, parse_dates=True)
    new_closes["OKTA"] = okta["Adj Close"]
    new_dollars["OKTA"] = okta.Close * okta.Volume
    current = pd.concat([current, pd.DataFrame([{"historical_symbol": "OKTA", "code": "OKTA", "CIK": OKTA_CIK,
                                                "sector": "information_technology"}])], ignore_index=True)
    dates = close.index.append(WEEK)
    new_close = pd.DataFrame(new_closes, index=dates)
    dollar = pd.DataFrame(new_dollars, index=dates)
    latest_adv = dollar.tail(20).mean().where(dollar.tail(20).count() == 20)
    latest_price = new_close.loc[SIGNAL]
    warm = new_close.tail(274).notna().all()
    new_volume = bars.Volume.loc[SIGNAL].copy()
    new_volume.loc["OKTA"] = okta.Volume.loc[SIGNAL]
    current["adv20"] = current.code.map(latest_adv)
    current["has_close"] = current.code.map(latest_price).gt(0)
    current["ready"] = current.code.map(warm).eq(True) & current.adv20.notna()
    current["eligible"] = (current.ready & current.adv20.ge(10_000_000)
                           & current.code.map(new_volume).gt(0) & current.sector.ne(""))
    current[["historical_symbol", "code", "CIK", "sector", "has_close", "ready", "eligible", "adv20"]].to_csv(
        out / "nov06_pool.csv", index=False)
    if min(current.has_close.mean(), current.ready.mean()) < .95:
        raise ValueError("Nov 6 coverage below 95%")
    spy_history = market["benchmark"]["Adj Close"]
    spy = pd.concat([spy_history, adjusted.loc[WEEK, "SPY"] * spy_history.at[OLD] / adjusted.at[OLD, "SPY"]])
    latest = ranked(SIGNAL, new_close, spy, current.loc[current.eligible])
    next_lists = {name: pick(latest, name, old[name]) for name in RECIPES}
    next_lists["rank_06_switch_from_baseline"] = pick(latest, "rank_06", last_baseline)
    targets = []
    for date, selections, ranks in [(OLD, old, old_rank), (SIGNAL, next_lists, latest)]:
        ranks = ranks.set_index("code")
        for name, codes in selections.items():
            recipe = "rank_06" if name.startswith("rank_06") else name
            previous = [] if date == OLD else old["baseline_70_30" if name.endswith("from_baseline") else recipe]
            targets.extend((date, name, code, ranks.at[code, "sector"], ranks.at[code, recipe], code in previous) for code in codes)
    pd.DataFrame(targets, columns=["signal", "recipe", "ticker", "sector", "score_rank", "retained"]).to_csv(
        out / "comparison_lists.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("download", "compare"))
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.stage == "download":
        download(args.out)
    else:
        compare(args.out)
