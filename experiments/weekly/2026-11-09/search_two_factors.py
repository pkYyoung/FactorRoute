"""Prespecified two-factor weekly grid; select using 2024–25, then review 2026."""
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from compare_strategies import HERE, ROOT, dump, load_config, relative_peers, sha, standardize


@dataclass(frozen=True)
class Rule:
    momentum: str
    weight: int
    holdings: int
    cap: int
    buffer: int


GRID = tuple(Rule(*v) for v in product(
    ("RSTR", "21d_peer_63d"), (50, 70, 85), (10, 15, 20), (0, 20, 30), (0, 5)
))
G0 = Rule("RSTR", 70, 10, 0, 0)
G2 = Rule("21d_peer_63d", 70, 10, 0, 0)
TRAIN_END = pd.Timestamp("2025-12-31")
FEE = .001
STRESS_FEE = .0015
DD_LIMIT = -.25


def inputs():
    cfg = load_config()
    root = ROOT / cfg["snapshot"]
    ranks_path = ROOT / cfg["rankings"]
    cal_path = root / "calendar_XNYS.csv"
    ranks = pd.read_csv(ranks_path, parse_dates=["date"])
    dates = pd.DatetimeIndex(pd.read_csv(cal_path).iloc[:, 0])
    dates = dates[dates <= pd.Timestamp(cfg["signal_date"])]
    ranks = ranks[ranks.date <= dates[-1]]
    expected = pd.Series(dates).groupby(dates.to_period("W-FRI")).max()
    if list(expected[expected >= ranks.date.min()]) != sorted(ranks.date.unique()):
        raise ValueError("Missing or unexpected archived signal dates")
    codes = sorted(set(ranks.code) | {"SPY"})
    files = [ranks_path, cal_path, HERE / "experiment_config.json",
             HERE / "compare_strategies.py", Path(__file__).resolve()]
    frames = {}
    for code in codes:
        path = root / "raw" / f"{code}.csv"
        files.append(path)
        frames[code] = pd.read_csv(path, parse_dates=["date"]).set_index("date").reindex(dates)
    close = pd.DataFrame({c: f["Adj Close"] for c, f in frames.items()}, index=dates)
    opening = pd.DataFrame({c: f.Open * f["Adj Close"] / f.Close for c, f in frames.items()}, index=dates)
    close = close.where(np.isfinite(close) & (close > 0))
    opening = opening.where(np.isfinite(opening) & (opening > 0))
    ranks = ranks.copy()
    order = {c: i for i, c in enumerate(codes)}
    ranks["symbol_id"] = ranks.code.map(order)
    ranks["sector_id"] = pd.factorize(ranks.sector, sort=True)[0]
    if (ranks.sector_id < 0).any():
        raise ValueError("Unknown sector in eligible pool")
    signals = []
    for date, group in ranks.groupby("date", sort=True):
        if group.code.duplicated().any() or group.issuer_id.duplicated().any():
            raise ValueError(f"Non-unique eligible company on {date}")
        i = dates.get_loc(date)
        history = close.loc[:date, group.code].tail(64)
        if len(history) != 64 or history.isna().any().any():
            raise ValueError(f"Incomplete 63-day history on {date}")
        recent = history.iloc[-1] / history.iloc[-22] - 1
        medium = history.iloc[-1] / history.iloc[0] - 1
        peer = relative_peers(recent, group.set_index("code").sector)
        # Keep the G2 relative/absolute momentum split at 4:3 when changing its total weight.
        combined = (4 * standardize(peer) + 3 * standardize(medium)) / 7
        signal = group.copy()
        signal["21d_peer_63d"] = signal.code.map(combined)
        if not np.isfinite(signal[["z_RSTR", "z_DASTD", "21d_peer_63d"]]).all().all():
            raise ValueError(f"Non-finite scores on {date}")
        signals.append((i, signal))
    return dates, codes, close.to_numpy(), opening.to_numpy(), signals, files


def build_rankings(signals):
    result = {}
    for kind, weight in sorted({(r.momentum, r.weight) for r in GRID}):
        momentum = "z_RSTR" if kind == "RSTR" else kind
        w = weight / 100
        rows = []
        for i, s in signals:
            # Preserve the published 70/30 and 40/30/30 arithmetic and tie breaks.
            score = w * s[momentum] - (.3 if weight == 70 else 1 - w) * s.z_DASTD
            ranked = s.assign(score=score).sort_values(["score", "code"], ascending=[False, True])
            rows.append((i, ranked.symbol_id.to_numpy(), ranked.sector_id.to_numpy()))
        result[kind, weight] = rows
    return result


def targets(rows, rule):
    """Retain incumbents still ranked inside N+buffer, then fill by score."""
    cap = rule.holdings * rule.cap // 100 if rule.cap else rule.holdings
    previous = set()
    holdings = []
    for i, symbols, sectors in rows:
        lead = min(len(symbols), rule.holdings + rule.buffer)
        retained = [j for j in range(lead) if symbols[j] in previous] if rule.buffer else []
        counts, chosen, chosen_set = {}, [], set()
        for j in (*retained, *range(len(symbols))):
            code, sector = symbols[j], sectors[j]
            if code in chosen_set or counts.get(sector, 0) >= cap:
                continue
            chosen.append(code)
            chosen_set.add(code)
            counts[sector] = counts.get(sector, 0) + 1
            if len(chosen) == rule.holdings:
                break
        if len(chosen) != rule.holdings:
            raise ValueError(f"Cannot fill target on signal index {i}: {rule}")
        previous = chosen_set
        holdings.append((i + 1, np.asarray(chosen, dtype=int)))
    return holdings


def simulate(holdings, close, opening, end, fee):
    """Return daily NAV and equity invested right after each rebalance."""
    if not holdings:
        raise ValueError("Empty schedule")
    nav = np.empty(end - holdings[0][0] + 1)
    invested_at, traded_fraction = [], []
    start = holdings[0][0]
    previous = np.array([], dtype=int)
    units = np.array([], dtype=float)
    for k, (i, symbols) in enumerate(holdings):
        until = holdings[k + 1][0] if k + 1 < len(holdings) else end + 1
        union = np.union1d(previous, symbols)
        px = opening[i, union]
        marks = close[i:until, symbols]
        if not np.isfinite(px).all() or not np.isfinite(marks).all():
            raise ValueError(f"Missing held price at rebalance index {i}")
        prior = np.zeros(len(union))
        at = np.searchsorted(union, symbols)
        prev_at = np.searchsorted(union, previous)
        prior[prev_at] = units * px[prev_at]
        capital = prior.sum() if k else 1.
        target = np.zeros(len(union))
        target[at] = 1 / len(symbols)
        invest = capital
        for _ in range(40):
            nxt = capital - fee * np.abs(invest * target - prior).sum()
            if abs(nxt - invest) <= 1e-14:
                invest = nxt
                break
            invest = nxt
        traded = np.abs(invest * target - prior).sum()
        if not np.isclose(invest + fee * traded, capital, atol=1e-11, rtol=0):
            raise ValueError("Rebalance budget does not reconcile")
        units = invest / len(symbols) / px[at]
        nav[i - start:until - start] = marks @ units
        invested_at.append(invest)
        traded_fraction.append(traded / capital)
        previous = symbols
    return nav, np.array(invested_at), np.array(traded_fraction)


def evaluate(schedule, close, opening, spy, dates, end, fee):
    nav, invested, turnover = simulate(schedule, close, opening, end, fee)
    entry = schedule[0][0]
    bench = close[entry:end + 1, spy] / opening[entry, spy] / (1 + fee)
    def drawdown(series):
        series = np.r_[1., series]
        return float(np.min(series / np.maximum.accumulate(series) - 1))
    windows = []
    for j in range(len(schedule) - 12):
        if j + 13 < len(schedule):
            last = schedule[j + 13][0] - 1
        elif dates[end].weekday() == 4:
            last = end
        else:
            break
        first = schedule[j][0]
        # Clone this week's target from cash, then follow the same precomputed targets.
        strategy = nav[last - entry] / invested[j] / (1 + fee) - 1
        benchmark = close[last, spy] / opening[first, spy] / (1 + fee) - 1
        windows.append((dates[first], dates[last], strategy, benchmark, strategy - benchmark))
    rolling = pd.DataFrame(windows, columns=["start", "end", "strategy", "SPY", "excess"])
    result = {
        "start": str(dates[entry].date()), "end": str(dates[end].date()),
        "net": float(nav[-1] - 1), "SPY": float(bench[-1] - 1),
        "excess": float(nav[-1] - bench[-1]), "drawdown": drawdown(nav),
        "SPY_drawdown": drawdown(bench), "rebalances": len(schedule),
        "mean_traded_fraction": float(turnover[1:].mean()),
        "rolling_windows": len(rolling),
        "rolling_wins": int((rolling.excess > 1e-12).sum()),
        "rolling_median": float(rolling.excess.median()),
        "rolling_worst": float(rolling.excess.min()),
    }
    return result, pd.DataFrame({"strategy": nav, "SPY": bench}, index=dates[entry:end + 1]), rolling


def main(out):
    if out.exists():
        raise FileExistsError(out)
    dates, codes, close, opening, signals, files = inputs()
    rankings = build_rankings(signals)
    spy = codes.index("SPY")
    split = int(dates.searchsorted(TRAIN_END, side="right")) - 1
    last = len(dates) - 1
    if not np.isfinite(close[:, spy]).all() or not np.isfinite(opening[:, spy]).all():
        raise ValueError("Incomplete SPY prices")
    train_schedules, review_schedules, train = {}, {}, {}
    for rule in GRID:
        rows = rankings[rule.momentum, rule.weight]
        train_schedules[rule] = targets([r for r in rows if r[0] + 1 <= split], rule)
        train[rule] = evaluate(train_schedules[rule], close, opening, spy, dates, split, FEE)[0]
    allowed = [r for r in GRID if train[r]["drawdown"] >= DD_LIMIT]
    if not allowed:
        raise ValueError("No training rule satisfies drawdown screen")
    winner = max(allowed, key=lambda r: (
        train[r]["rolling_median"], train[r]["excess"], -train[r]["mean_traded_fraction"]
    ))
    # The choice above is complete before any post-cutoff returns are scored.
    review = {}
    for rule in GRID:
        rows = rankings[rule.momentum, rule.weight]
        review_schedules[rule] = targets([r for r in rows if split < r[0] + 1 <= last], rule)
        review[rule] = evaluate(review_schedules[rule], close, opening, spy, dates, last, FEE)[0]

    reference = ROOT / load_config()["rankings"]
    ref_nav = reference.with_name("daily_nav.csv")
    files.append(ref_nav)
    base_rows = rankings[G0.momentum, G0.weight]
    full_base = targets([r for r in base_rows if r[0] < last], G0)
    _, g0_nav, _ = evaluate(full_base, close, opening, spy, dates, last, FEE)
    archived = pd.read_csv(ref_nav, index_col=0, parse_dates=True)
    np.testing.assert_allclose(g0_nav.strategy, archived.loc[g0_nav.index, "strategy"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(g0_nav.SPY, archived.loc[g0_nav.index, "SPY"], atol=1e-10, rtol=0)
    published = pd.read_csv(HERE / "selected.csv")
    files.append(HERE / "selected.csv")
    for name, rule in (("G0", G0), ("G2", G2)):
        all_rows = rankings[rule.momentum, rule.weight]
        picks = targets(all_rows, rule)[-1][1]
        if [codes[j] for j in picks] != published.loc[published.group == name, "code"].tolist():
            raise ValueError(f"{name} does not reproduce the October 30 list")

    out.mkdir(parents=True)
    grid = pd.DataFrame([
        {**asdict(r), "passes_train_dd": r in allowed, "chosen_on_train": r == winner,
         **{f"train_{k}": v for k, v in train[r].items()},
         **{f"review_{k}": v for k, v in review[r].items()}} for r in GRID
    ])
    grid.to_csv(out / "grid.csv", index=False)
    summary = {}
    for label, rule in (("candidate", winner), ("G0", G0), ("G2", G2)):
        full_rows = rankings[rule.momentum, rule.weight]
        full = targets([r for r in full_rows if r[0] < last], rule)
        summary[label] = {"rule": asdict(rule)}
        for period, schedule, end in (
            ("train", train_schedules[rule], split), ("review", review_schedules[rule], last),
            ("full", full, last),
        ):
            result, nav, rolling = evaluate(schedule, close, opening, spy, dates, end, FEE)
            stress = evaluate(schedule, close, opening, spy, dates, end, STRESS_FEE)[0]
            summary[label][period] = {**result, "excess_15bps": stress["excess"]}
            nav.to_csv(out / f"{label}_{period}_nav.csv", index_label="date")
            rolling.to_csv(out / f"{label}_{period}_13weeks.csv", index=False)
        if label == "candidate":
            pd.DataFrame([
                (dates[i - 1], dates[i], codes[j]) for i, ids in full for j in ids
            ], columns=["signal", "entry", "code"]).to_csv(out / "candidate_history.csv", index=False)
    dump(out / "metrics.json", {
        "grid_size": len(GRID), "train_drawdown_screen": DD_LIMIT,
        "eligible_rules": len(allowed), "review_winners": sum(r["excess"] > 0 for r in review.values()),
        "selection": "maximize training 13-week median excess subject to drawdown screen",
        "train_cutoff": str(TRAIN_END.date()), "fee": FEE, "strategies": summary,
    })
    dump(out / "input_sha256.json", {str(p.relative_to(ROOT)): sha(p) for p in files})
    print((out / "metrics.json").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out.resolve())
