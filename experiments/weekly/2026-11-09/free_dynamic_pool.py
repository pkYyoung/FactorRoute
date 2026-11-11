"""2020 free-data pilot with dated membership, issuer mapping and explicit gaps."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

import search_two_factors as grid
from compare_strategies import ROOT, dump, sha, standardize
from extend_price_history import RECIPES
from factorcode.data.factors import BetaFactor, MomentumFactor, ResidualVolatilityFactor
from factorcode.data.yfinance_source import load_snapshot
from test_subindustry_cap import RULE


INPUT = ROOT / "outputs/weekly/2026-11-09/free-pit-pilot"
SNAPSHOT = ROOT / "data/yfinance/extended_2015_to_20261030"


def history(path):
    f = pd.read_csv(path, dtype=str).fillna("")
    f["symbol"] = f.symbol.str.replace(".", "-", regex=False)
    f["approximate_date"] = f.date_added.str.endswith("*") | f.date_removed.str.endswith("*")
    f["added"] = pd.to_datetime(f.date_added.str.rstrip("*").replace("", "1900-01-01"))
    f["removed"] = pd.to_datetime(f.date_removed.str.rstrip("*").replace("", "2262-01-01"))
    f["seen"] = pd.to_datetime(f.created_at.replace("", "1900-01-01"))
    # Official effective dates override the community page's observation lag.
    for symbol, cik, effective, fields in (
        ("ETSY", "0001370637", "2020-09-21", ("seen",)),
        ("POOL", "0000945841", "2020-10-07", ("seen",)),
        ("CTL", "0000018926", "2020-09-18", ("removed",)),
        ("LUMN", "0000018926", "2020-09-18", ("seen",)),
    ):
        row = (f.symbol == symbol) & (f.cik == cik)
        if row.sum() != 1:
            raise ValueError(f"Official correction does not map uniquely: {symbol}")
        f.loc[row, list(fields)] = pd.Timestamp(effective)
    return f


def members_on(hist, signal):
    return hist.loc[(hist.added <= signal) & (hist.removed > signal) & (hist.seen <= signal)].copy()


def resolve(row, issuer_counts, original, by_issuer, sec, available, include_extra):
    if row.symbol in original and original[row.symbol] == row.cik:
        return row.symbol, "current_exact_cik"
    if not row.cik:
        return None, "missing_cik"
    tickers = sec.get(row.cik, ())
    if include_extra and row.symbol in tickers and row.symbol in available and row.symbol not in original:
        return row.symbol, "SEC_exact_cik"
    # A company-level CIK cannot disambiguate concurrent share classes.
    if issuer_counts[row.cik] != 1:
        return None, "ambiguous_share_class"
    if len(by_issuer.get(row.cik, ())) == 1:
        return by_issuer[row.cik][0], "current_cik_alias"
    if not include_extra:
        return None, "not_in_frozen_prices"
    if len(tickers) == 1 and tickers[0] in available and tickers[0] not in original:
        return tickers[0], "SEC_cik_alias"
    return None, "no_verified_history"


def extra_prices(directory, dates, market):
    closes, opens, volume, dollar = {}, {}, {}, {}
    accepted = []
    for path in sorted(directory.glob("*.csv")):
        if path.stem in market["close"].columns:
            continue
        f = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
        if (f.empty or f.index.duplicated().any() or not f.index.isin(dates).all()
                or not {"Open", "Close", "Adj Close", "Volume"} <= set(f.columns)):
            continue
        cols = f[["Open", "Close", "Adj Close", "Volume"]]
        if not np.isfinite(cols).all().all() or (cols.iloc[:, :3] <= 0).any().any() or (f.Volume < 0).any():
            continue
        closes[path.stem] = f["Adj Close"]
        opens[path.stem] = f.Open * f["Adj Close"] / f.Close
        volume[path.stem] = f.Volume
        dollar[path.stem] = f.Close * f.Volume
        accepted.append(path)
    return [pd.concat([market[k], pd.DataFrame(v, index=dates)], axis=1)
            for k, v in zip(("close", "open", "volume", "dollar"),
                            (closes, opens, volume, dollar), strict=True)], accepted


def scores(closes, bench, end, names):
    # No forward filling or bridging unquoted sessions in the signal window.
    r = closes.iloc[end - 273:end + 1][names].pct_change(fill_method=None).iloc[1:].to_numpy()
    x = bench.iloc[end - 251:end + 1].to_numpy()
    if r.shape != (273, len(names)) or not np.isfinite(r).all():
        raise ValueError("Invalid factor window")
    w = BetaFactor.W_HBETA
    y = r[-252:]
    my, mx = (w @ y) / w.sum(), (w @ x) / w.sum()
    b = (w * (x - mx)) @ (y - my) / (np.sum(w * (x - mx)**2) + 1e-18)
    v = ResidualVolatilityFactor.W_DASTD
    vy = (v @ y) / v.sum()
    sd = np.sqrt((v @ (y - vy)**2) / v.sum())
    lagged = np.convolve(MomentumFactor.W_RSTR, np.ones(11) / 11)
    rstr = lagged @ np.log1p(np.maximum(r[:-11], -.9999))
    return pd.DataFrame({"RSTR": rstr, "DASTD": sd, "HALPHA": my - b * mx, "HBETA": b}, index=names)


def main(out):
    if out.exists():
        raise FileExistsError(out)
    hist = history(INPUT / "components_history.csv")
    for d, ticker, expected in [("2020-04-03", "M", True), ("2020-04-06", "M", False),
                                 ("2020-09-17", "CTL", True), ("2020-09-18", "CTL", False),
                                 ("2020-09-18", "LUMN", True), ("2020-09-21", "ETSY", True),
                                 ("2020-10-06", "ETFC", True), ("2020-10-07", "ETFC", False),
                                 ("2020-10-07", "POOL", True),
                                 ("2020-12-18", "TSLA", False), ("2020-12-21", "TSLA", True)]:
        assert (ticker in set(members_on(hist, pd.Timestamp(d)).symbol)) == expected, (d, ticker)
    market = load_snapshot(SNAPSHOT, "2026-10-30")
    dates = market["close"].index
    (closes, opens, volume, dollar), bars = extra_prices(INPUT / "yahoo", dates, market)
    adv = dollar.rolling(20, min_periods=20).mean()
    warm = closes.notna().rolling(274, min_periods=274).sum().eq(274)
    bench = market["benchmark"]
    bench_ret = bench["Adj Close"].pct_change(fill_method=None)
    original = market["universe"].set_index("code").issuer_id.to_dict()
    by_issuer = {k: list(v.code) for k, v in market["universe"].groupby("issuer_id")}
    sec_paths = sorted((INPUT / "sec").glob("*.json"))
    sec = {}
    for path in sec_paths:
        r = json.loads(path.read_text())
        sec[r["cik"]] = [t.replace(".", "-") for t in r["tickers"]]
    codes = sorted(set(closes.columns) | {"SPY"})
    code_id = {c: i for i, c in enumerate(codes)}
    sector_id = {s: i for i, s in enumerate(sorted(set(hist.sector) - {""}))}
    available = set(closes.columns)
    sessions = pd.Series(dates).groupby(dates.to_period("W-FRI")).max()
    start = dates.searchsorted("2020-01-01")
    end = dates.searchsorted("2020-12-31", side="right") - 1
    audit, coverage = [], []
    rank_rows = {k: {n: [] for n in ("baseline_70_30", "alpha_40_40_20")} for k in
                 ("existing_prices", "augmented_prices")}
    for signal in sessions:
        j = dates.get_loc(signal)
        if j < 273 or signal < pd.Timestamp("2016-01-01"):
            continue
        members = members_on(hist, signal)
        counts = Counter(members.cik)
        run_pilot = start <= j + 1 <= end
        for mode in rank_rows:
            if mode == "existing_prices" and not run_pilot:
                continue
            candidates = []
            resolved = 0
            for row in members.itertuples():
                code, source = resolve(row, counts, original, by_issuer, sec, available,
                                       mode == "augmented_prices")
                if code not in available:
                    status = "unresolved_identity_or_source"
                elif not np.isfinite(closes.at[signal, code]):
                    status = "no_signal_close"
                else:
                    resolved += 1
                    if not row.sector:
                        status = "missing_sector"
                    elif not warm.at[signal, code]:
                        status = "missing_warmup_or_quote"
                    elif not (adv.at[signal, code] >= 10_000_000 and volume.at[signal, code] > 0):
                        status = "liquidity_filter"
                    else:
                        status = "eligible"
                        candidates.append((code, row.symbol, row.cik, row.sector, adv.at[signal, code]))
                audit.append((signal, mode, row.symbol, row.cik, code, source, status, row.approximate_date))
            # Membership eligibility is applied before issuer de-duplication or standardization.
            pool = pd.DataFrame(candidates, columns=["code", "historical", "cik", "sector", "adv"])
            pool = pool.sort_values(["adv", "code"], ascending=[False, True]).drop_duplicates("cik")
            if pool.code.duplicated().any():
                raise ValueError(f"Unresolved identity collision on {signal}")
            coverage.append((signal, mode, len(members), resolved, len(pool)))
            if not run_pilot:
                continue
            if len(pool) < 100:
                raise ValueError(f"Insufficient qualified members on {signal}")
            factors = scores(closes, bench_ret, j, pool.code.tolist())
            z = np.column_stack([standardize(factors[k]) * sign for k, sign in
                                 zip(("RSTR", "DASTD", "HALPHA", "HBETA"), (1, -1, 1, -1), strict=True)])
            ids = pool.code.map(code_id).to_numpy()
            sectors = pool.sector.map(sector_id).to_numpy()
            for name in rank_rows[mode]:
                order = np.lexsort((ids, -(z @ np.asarray(RECIPES[name]))))
                rank_rows[mode][name].append((j, ids[order], sectors[order]))

    out.mkdir(parents=True)
    cover = pd.DataFrame(coverage, columns=["signal", "pool", "members", "resolved_with_close", "eligible"])
    cover.to_csv(out / "weekly_coverage.csv", index=False)
    pd.DataFrame(audit, columns=["signal", "pool", "historical_symbol", "CIK", "quote_code",
                                 "mapping", "status", "approximate_date"]).to_csv(
        out / "member_mapping_and_gaps.csv", index=False)
    cl = closes.assign(SPY=bench["Adj Close"])[codes].to_numpy()
    op = opens.assign(SPY=bench.Open * bench["Adj Close"] / bench.Close)[codes].to_numpy()
    metrics = {}
    for mode, group in rank_rows.items():
        metrics[mode] = {}
        for name, rows in group.items():
            targets = grid.targets(rows, RULE)
            picks = []
            missing = []
            for k, (i, ids) in enumerate(targets):
                until = targets[k + 1][0] if k + 1 < len(targets) else end + 1
                for id_ in ids:
                    picks.append((dates[i - 1], dates[i], codes[id_]))
                    bad_close = dates[i:until][~np.isfinite(cl[i:until, id_])]
                    for date in bad_close:
                        missing.append((date, codes[id_], "held_close"))
                    for at, kind in ((i, "entry_open"), (until, "next_rebalance_open")):
                        if at <= end and not np.isfinite(op[at, id_]):
                            missing.append((dates[at], codes[id_], kind))
            tag = f"{mode}_{name}"
            pd.DataFrame(picks, columns=["signal", "entry", "code"]).to_csv(out / f"{tag}_picks.csv", index=False)
            if missing:
                pd.DataFrame(missing, columns=["date", "code", "field"]).to_csv(
                    out / f"{tag}_unresolved_exits.csv", index=False)
                metrics[mode][name] = {"status": "unresolved_held_quote", "missing_marks": len(missing)}
                continue
            m, nav, rolling = grid.evaluate(targets, cl, op, code_id["SPY"], dates, end, grid.FEE)
            metrics[mode][name] = {"status": "completed_with_partial_coverage", **m}
            nav.to_csv(out / f"{tag}_nav.csv", index_label="date")
            rolling.to_csv(out / f"{tag}_13weeks.csv", index=False)
    dump(out / "metrics.json", metrics)
    dump(out / "inputs.json", {str(p.relative_to(ROOT)): sha(p) for p in
                              [Path(__file__).resolve(), INPUT / "components_history.csv",
                               SNAPSHOT / "manifest.json", *sec_paths, *bars]})
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    main(parser.parse_args().out.resolve())
