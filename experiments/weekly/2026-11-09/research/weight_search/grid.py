"""Prepare the nine-factor panel and evaluate frozen weekly portfolios."""

from concurrent.futures import ThreadPoolExecutor
import ctypes
from dataclasses import dataclass
from time import monotonic

import numpy as np
import pandas as pd

from ..data_coverage.audit import build_coverage, summarize, pit
from . import search_daily_factors as daily
from . import search_two_factors as grid


START = pd.Timestamp("2023-03-17")
END = pd.Timestamp("2026-10-30")
SPLIT = pd.Timestamp("2025-12-31")
WIDTH = 13
BASE = daily.BASE
ALPHA = (40, 0, 40, 0, 20, 0, 0, 0, 0)
REFERENCES = {"SPY": None, "baseline_70_30": BASE, "original_G0": BASE, "alpha_40_40_20": ALPHA}
NAMES = daily.NAMES


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    signals: pd.DatetimeIndex
    close: np.ndarray
    opening: np.ndarray
    factors: np.ndarray
    ids: np.ndarray
    sectors: np.ndarray
    counts: np.ndarray
    sector_count: int
    entries: np.ndarray
    ends: np.ndarray
    marks: np.ndarray
    overnight: np.ndarray

    @property
    def days(self):
        return (self.ends - self.entries + 1).astype(np.int32)


def load_panel(source, out):
    market, audit, histories, _, _ = build_coverage(source)
    coverage, start = summarize(audit, END)
    if start != START:
        raise ValueError(f"Numeric audit start changed to {start}")
    audit = audit.loc[audit.signal.between(START, END)].copy()
    dates = market["close"].index
    weeks = pd.DatetimeIndex(sorted(audit.signal.unique()))
    indexes = dates.get_indexer(weeks)
    if len(weeks) != 190 or weeks[-1] != END or (indexes < 0).any():
        raise ValueError("Unexpected approved window calendar")
    audit = audit.loc[audit.signal < END].copy()
    audit["identity"] = list(zip(audit.historical_symbol, audit.CIK, audit.quote_code, strict=True))
    audit["security"] = [((row.quote_code if row.price_source == "archived" else row.historical_symbol),
                          row.CIK, row.price_source) for row in audit.itertuples()]
    # A confirmed regular-listing warm-up exception is for coverage. Eligibility still uses actual bars.
    active = audit.loc[audit.liquid].copy()
    lookup = active.drop_duplicates("security").set_index("security").identity.to_dict()
    names = sorted(lookup)
    ids = {key: i for i, key in enumerate(names)}
    stock_closes, stock_opens = [], []
    for key in names:
        cl, op, vol = histories[lookup[key]]
        stock_closes.append(cl.to_numpy())
        stock_opens.append(op.where(vol > 0).to_numpy())
    spy = market["benchmark"]
    close = np.column_stack([*stock_closes, spy["Adj Close"]])
    opening = np.column_stack([*stock_opens, spy.Open * spy["Adj Close"] / spy.Close])
    spy_id = len(names)
    month_dates = pd.Series(dates).groupby(dates.to_period("M")).max()
    month_indices = dates.get_indexer(month_dates)
    sector_ids = {name: i for i, name in enumerate(sorted(active.sector.unique()))}
    # CMRA needs the actual month-end marks, which can extend beyond the daily warm-up.
    factor_coverage, factor_rows = [], []
    for signal, row in audit.groupby("signal"):
        i = dates.get_loc(signal)
        month_ends = month_indices[month_indices <= i][-13:]
        eligible = row.loc[row.liquid].copy()
        eligible["id"] = eligible.security.map(ids)
        complete = np.isfinite(close[np.ix_(month_ends, eligible.id.to_numpy())]).all(axis=0)
        missing = eligible.loc[~complete, "identity"]
        raw_ready = row.factor_ready & ~row.identity.isin(missing)
        denominator = ~row.verified_short_listing
        common_rate = raw_ready[denominator].mean()
        eligible = eligible.loc[complete].sort_values(["adv20", "id"], ascending=[False, True])
        eligible = eligible.drop_duplicates("CIK").sort_values("id")
        available = np.isfinite(opening[i + 1, eligible.id]).mean()
        factor_coverage.append((signal, common_rate, available, len(eligible), " ".join(x[0] for x in missing)))
        symbols = eligible.id.to_numpy()
        close_frame = pd.DataFrame(close[:, symbols], columns=symbols, index=dates)
        reference = pit.scores(close_frame, spy["Adj Close"].pct_change(fill_method=None), i, symbols.tolist())
        sectors = eligible.sector.map(sector_ids).to_numpy()
        factor_rows.append((i, pd.DataFrame({"symbol_id": symbols, "code": symbols, "sector": sectors, "sector_id": sectors,
                                             "HBETA": reference.HBETA.to_numpy(),
                                             "z_RSTR": pit.standardize(reference.RSTR).to_numpy(),
                                             "z_DASTD": pit.standardize(reference.DASTD).to_numpy()})))
    observed = pd.DataFrame(factor_coverage, columns=["signal", "nine_signal_coverage", "next_open_coverage",
                                                      "common_pool", "extra_missing_cmra"])
    observed.to_csv(out / "common_pool_coverage.csv", index=False)
    if not (observed.nine_signal_coverage >= .95).all() or not (observed.next_open_coverage >= .98).all():
        raise ValueError("At least one signal week does not pass the nine-signal coverage gate")
    signals = daily.factor_panels(dates, close, spy_id, factor_rows)
    counts = np.array([len(row[1]) for row in signals], dtype=np.int32)
    factors = np.zeros((len(signals), counts.max(), len(NAMES)))
    securities = np.full(factors.shape[:2], -1, dtype=np.int32)
    sectors = np.full_like(securities, -1)
    for k, (_, ticker_ids, industry_ids, values) in enumerate(signals):
        factors[k, :counts[k]] = values
        securities[k, :counts[k]] = ticker_ids
        sectors[k, :counts[k]] = industry_ids
    entries = indexes[:-1] + 1
    ends = indexes[1:]
    if np.any(ends - entries >= 5):
        raise ValueError("Expected at most five exchange sessions per signal week")
    marks = np.full((len(signals), 5, close.shape[1]), np.nan)
    overnight = np.full((len(signals), close.shape[1]), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        for w, (entry, end) in enumerate(zip(entries, ends, strict=True)):
            marks[w, :end - entry + 1] = close[entry:end + 1] / opening[entry]
            if w:
                overnight[w] = opening[entry] / opening[entries[w - 1]]
    return Panel(dates, weeks, close, opening, factors, securities, sectors, counts,
                 len(sector_ids), entries, ends, marks, overnight)


def windows(panel, period):
    starts = np.arange(len(panel.entries) - WIDTH + 1, dtype=np.int32)
    begin = panel.dates[panel.entries[starts]]
    finish = panel.dates[panel.ends[starts + WIDTH - 1]]
    if period == "training":
        starts = starts[finish <= SPLIT]
    elif period == "review":
        starts = starts[begin > SPLIT]
    return pd.DataFrame({"index": starts, "start": panel.dates[panel.entries[starts]],
                         "end": panel.dates[panel.ends[starts + WIDTH - 1]],
                         "nonoverlap": (starts - starts[0]) % WIDTH == 0})


def native_engine(panel, binary):
    pointer = ctypes.c_void_p
    int_ = ctypes.c_int
    function = ctypes.CDLL(str(binary.resolve())).evaluate_grid
    function.argtypes = [pointer] * 4 + [int_] * 4 + [pointer] * 4 + [int_, pointer, int_, int_, ctypes.c_double, pointer]
    function.restype = int_
    inputs = [panel.factors, panel.ids, panel.sectors, panel.counts, panel.marks, panel.overnight, panel.days]
    data = [x.ctypes.data for x in inputs]

    def run(weights, starts, fee, original=False):
        output = np.empty((len(starts), 3))
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        starts = np.ascontiguousarray(starts, dtype=np.int32)
        result = function(*data[:4], panel.ids.shape[1], int(starts[-1]) + WIDTH, panel.sector_count, panel.close.shape[1],
                          *data[4:], starts.ctypes.data, len(starts), weights.ctypes.data,
                          10 if original else 3, not original, fee, output.ctypes.data)
        if result:
            raise RuntimeError(f"Native simulation error {result}")
        return output

    return run


def spy_metrics(panel, starts, fee):
    result = []
    for start in starts:
        entry = panel.entries[start]
        end = panel.ends[start + WIDTH - 1]
        nav = panel.close[entry:end + 1, -1] / panel.opening[entry, -1] / (1 + fee)
        path = np.r_[1., nav]
        result.append((nav[-1] - 1, (1 - path / np.maximum.accumulate(path)).max(), 0.0))
    return np.asarray(result)


def validate(panel, run):
    starts = np.array([0, 1, 18, 72, 120, 150, 176], dtype=np.int32)
    for weights in (BASE, ALPHA, daily.G2, (69, 31, 0, 0, 0, 0, 0, 0, 0)):
        rows = []
        for week, n in enumerate(panel.counts):
            ids = panel.ids[week, :n]
            sectors = panel.sectors[week, :n]
            order = np.lexsort((ids, -(panel.factors[week, :n] @ weights)))
            rows.append((panel.entries[week] - 1, ids[order], sectors[order]))
        for original in (False, True):
            for fee in (.001, .0015):
                observed = run(weights, starts, fee, original)
                for j, start in enumerate(starts):
                    schedule = grid.targets(rows[start:start + WIDTH], grid.G0 if original else daily.RULE)
                    try:
                        nav, _, turnover = grid.simulate(schedule, panel.close, panel.opening, panel.ends[start + WIDTH - 1], fee)
                    except ValueError:
                        assert np.isnan(observed[j]).all()
                        continue
                    path = np.r_[1., nav]
                    expected = [nav[-1] - 1, (1 - path / np.maximum.accumulate(path)).max(), turnover[1:].mean()]
                    np.testing.assert_allclose(observed[j], expected, atol=2e-11, rtol=0)


def compute(run, weights, starts, workers, progress=True):
    chunks = [(i, min(i + 64, len(weights))) for i in range(0, len(weights), 64)]
    values = np.empty((len(weights), len(starts), 3))
    started = monotonic()

    def batch(bounds):
        first, last = bounds
        for i in range(first, last):
            values[i] = run(weights[i], starts, .001)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(batch, chunks), 1):
            if progress and (i % 40 == 0 or i == len(chunks)):
                print(f"Evaluated {min(i * 64, len(weights)):,}/{len(weights):,} recipes in {monotonic()-started:.1f}s", flush=True)
    return values


def summarize_grid(weights, values, schedule, spy, baseline, common):
    info = pd.DataFrame(weights, columns=NAMES)
    for label, mask in (("rolling", common), ("disjoint", common & schedule.nonoverlap.to_numpy())):
        sample = values[:, mask]
        excess = sample[:, :, 0] - spy[None, mask, 0]
        diff = sample[:, :, 0] - baseline[None, mask, 0]
        dd_delta = sample[:, :, 1] - baseline[None, mask, 1]
        info[f"{label}_n"] = mask.sum()
        info[f"{label}_spy_wins"] = (excess > 1e-12).sum(axis=1)
        info[f"{label}_baseline_wins"] = (diff > 1e-12).sum(axis=1)
        info[f"{label}_median_excess_spy"] = np.median(excess, axis=1)
        info[f"{label}_median_excess_baseline"] = np.median(diff, axis=1)
        info[f"{label}_median_dd_delta"] = np.median(dd_delta, axis=1)
        info[f"{label}_median_dd"] = np.median(sample[:, :, 1], axis=1)
        info[f"{label}_worst_dd"] = sample[:, :, 1].max(axis=1)
        info[f"{label}_worst_net"] = sample[:, :, 0].min(axis=1)
        info[f"{label}_worst_excess_spy"] = excess.min(axis=1)
        info[f"{label}_median_turnover"] = np.median(sample[:, :, 2], axis=1)
    base_worst = baseline[common & schedule.nonoverlap.to_numpy(), 1].max()
    info["passes_training_filter"] = ((info.disjoint_median_excess_spy > 0)
        & (info.disjoint_median_excess_baseline > 0) & (info.disjoint_median_dd_delta <= 1e-12)
        & (info.disjoint_worst_dd <= base_worst + 1e-12))
    columns = ["passes_training_filter", "disjoint_median_excess_spy", "rolling_median_excess_spy", "disjoint_worst_dd", *NAMES]
    return info.sort_values(columns, ascending=[False, False, False, True, *([True] * len(NAMES))])


def review(out, panel, run, training_common, shortlist):
    top = pd.read_csv(out / shortlist)
    picks = {f"rank_{row.training_rank:02}": tuple(getattr(row, name) for name in NAMES)
             for row in top.itertuples(index=False)}
    all_rows, summaries = [], []
    for period in ("training", "review", "full"):
        schedule = windows(panel, period)
        starts = schedule["index"].to_numpy()
        paired = []
        for fee in (.001, .0015):
            spy = spy_metrics(panel, starts, fee)
            returns = {name: spy if weights is None else run(weights, starts, fee, name == "original_G0")
                       for name, weights in {**REFERENCES, **picks}.items()}
            paired.append((fee, returns))
        common = np.isfinite([values for _, frame in paired for values in frame.values()]).all(axis=(0, 2))
        if period == "training":
            common &= training_common
        for fee, returns in paired:
            baseline = returns["baseline_70_30"]
            spy = returns["SPY"]
            for name, values in returns.items():
                frame = schedule.assign(period=period, fee_bps=round(fee * 10_000), recipe=name, common=common,
                                        net=values[:, 0], drawdown=values[:, 1], turnover=values[:, 2],
                                        excess_spy=values[:, 0] - spy[:, 0], excess_baseline=values[:, 0] - baseline[:, 0],
                                        dd_delta_baseline=values[:, 1] - baseline[:, 1])
                all_rows.append(frame)
                for label, mask in (("rolling", common), ("disjoint", common & schedule.nonoverlap.to_numpy())):
                    cut = frame.loc[mask]
                    summaries.append({"period": period, "window_type": label, "fee_bps": round(fee * 10_000),
                                      "recipe": name, "windows": len(cut), "spy_wins": int((cut.excess_spy > 1e-12).sum()),
                                      "baseline_wins": int((cut.excess_baseline > 1e-12).sum()),
                                      "median_net": cut.net.median(), "median_excess_spy": cut.excess_spy.median(),
                                      "median_excess_baseline": cut.excess_baseline.median(), "median_drawdown": cut.drawdown.median(),
                                      "median_dd_delta_baseline": cut.dd_delta_baseline.median(), "worst_drawdown": cut.drawdown.max(),
                                      "worst_net": cut.net.min(), "worst_excess_spy": cut.excess_spy.min(),
                                      "median_turnover": cut.turnover.median()})
    pd.concat(all_rows, ignore_index=True).to_csv(out / "finalist_windows.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "finalist_summary.csv", index=False)
