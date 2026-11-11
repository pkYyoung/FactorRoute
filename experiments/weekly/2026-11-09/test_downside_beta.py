"""Isolate one change: ex-ante portfolio downside beta at most 1.0."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import search_two_factors as grid
from compare_strategies import ROOT, dump, sha
from test_subindustry_cap import RULE, weekly_summary


LOOKBACK = 126
MIN_DOWN = 30
BETA_LIMIT = 1.0


def downside_beta(close, spy, rows):
    returns = close[1:] / close[:-1] - 1
    result = {}
    for i, symbols, _ in rows:
        window = returns[i - LOOKBACK:i]
        if len(window) != LOOKBACK or not np.isfinite(window[:, spy]).all():
            raise ValueError(f"Missing SPY risk history at {i}")
        down = window[:, spy] < 0
        x = window[down, spy]
        y = window[down][:, symbols]
        if len(x) < MIN_DOWN or not np.isfinite(y).all():
            raise ValueError(f"Insufficient downside observations at {i}")
        x = x - x.mean()
        denom = x @ x
        if denom <= 0:
            raise ValueError(f"Zero downside variance at {i}")
        result[i] = (x @ (y - y.mean(axis=0))) / denom
    return result


def min_completion(beta_order, available, sectors, betas, counts, remaining):
    """Cheapest feasible completion in beta units under the sector quotas."""
    cost = 0.
    counts = counts.copy()
    for j in beta_order:
        if remaining == 0:
            break
        sector = sectors[j]
        if available[j] and counts[sector] < 3:
            counts[sector] += 1
            cost += betas[j]
            remaining -= 1
    return cost if not remaining else np.inf


def guarded_targets(rows, risk, limit=BETA_LIMIT):
    previous = set()
    result = []
    for i, codes, sectors in rows:
        beta = risk[i]
        preferred = [j for j in range(15) if codes[j] in previous]
        preferred_set = set(preferred)
        order = preferred + [j for j in range(len(codes)) if j not in preferred_set]
        beta_order = np.argsort(beta, kind="stable")
        available = np.ones(len(codes), dtype=bool)
        counts, chosen = Counter(), []
        total = 0.
        for j in order:
            available[j] = False
            sector = sectors[j]
            if counts[sector] >= 3:
                continue
            counts[sector] += 1
            needed = 9 - len(chosen)
            lower_bound = min_completion(beta_order, available, sectors, beta, counts, needed)
            if total + beta[j] + lower_bound <= 10 * limit + 1e-12:
                chosen.append(codes[j])
                total += beta[j]
                if len(chosen) == 10:
                    break
            else:
                counts[sector] -= 1
        if len(chosen) != 10:
            raise ValueError(f"Beta cap is infeasible at signal {i}; no cap relaxation")
        previous = set(chosen)
        result.append((i + 1, np.array(chosen, dtype=int)))
    return result


def main(out):
    if out.exists():
        raise FileExistsError(out)
    dates, codes, close, opening, signals, files = grid.inputs()
    rows = grid.build_rankings(signals)["RSTR", 70]
    spy = codes.index("SPY")
    risk = downside_beta(close, spy, rows)
    # Unbounded risk must reproduce the original industry-and-buffer selection.
    unrestricted = guarded_targets(rows, risk, np.inf)
    assert all(np.array_equal(a, b) for (_, a), (_, b) in zip(
        unrestricted, grid.targets(rows, RULE), strict=True
    ))
    by_day = {i: (dict(zip(ids, risk[i], strict=True)), dict(zip(ids, sect, strict=True)))
              for i, ids, sect in rows}
    last = len(dates) - 1
    split = dates.searchsorted(grid.TRAIN_END, side="right") - 1
    out.mkdir(parents=True)
    summary = {}
    audit = []
    for period, rs in (("full", [r for r in rows if r[0] < last]),
                       ("review", [r for r in rows if split < r[0] + 1 <= last])):
        summary[period] = {}
        for label, plan in (("baseline", grid.targets(rs, RULE)), ("beta_cap", guarded_targets(rs, risk))):
            estimates = []
            for i, ids in plan:
                betas, sectors = by_day[i - 1]
                estimate = float(np.mean([betas[j] for j in ids]))
                if label == "beta_cap":
                    assert len(set(ids)) == 10
                    assert max(Counter(sectors[j] for j in ids).values()) <= 3
                    assert estimate <= BETA_LIMIT + 1e-12
                estimates.append(estimate)
                audit.extend((period, label, dates[i - 1], dates[i], codes[j], betas[j], estimate)
                             for j in ids)
            m, nav, rolling = grid.evaluate(plan, close, opening, spy, dates, last, grid.FEE)
            stress = grid.evaluate(plan, close, opening, spy, dates, last, grid.STRESS_FEE)[0]
            w, weekly = weekly_summary(nav)
            summary[period][label] = {
                **m, "weekly": w, "excess_15bps": stress["excess"],
                "mean_estimated_beta": float(np.mean(estimates)),
                "max_estimated_beta": max(estimates),
                "signals_above_1": sum(e > 1 + 1e-12 for e in estimates),
            }
            nav.to_csv(out / f"{period}_{label}_nav.csv", index_label="date")
            weekly.to_csv(out / f"{period}_{label}_weekly.csv", index_label="week_end")
            rolling.to_csv(out / f"{period}_{label}_13weeks.csv", index=False)
    latest = {}
    for label, plan in (("baseline", grid.targets(rows, RULE)), ("beta_cap", guarded_targets(rows, risk))):
        i, ids = plan[-1]
        betas, _ = by_day[i - 1]
        latest[label] = {"codes": [codes[j] for j in ids],
                         "estimated_beta": float(np.mean([betas[j] for j in ids]))}
    summary["latest_signal"] = latest
    summary["risk_definition"] = {"lookback": LOOKBACK, "minimum_down_days": MIN_DOWN, "cap": BETA_LIMIT}
    pd.DataFrame(audit, columns=["period", "rule", "signal", "entry", "code",
                                 "downside_beta", "portfolio_downside_beta"]).to_csv(out / "picks.csv", index=False)
    dump(out / "metrics.json", summary)
    dump(out / "input_sha256.json", {str(p.relative_to(ROOT)): sha(p) for p in
                                     [*files, Path(__file__).resolve(),
                                      Path(__file__).with_name("test_subindustry_cap.py").resolve()]})
    print((out / "metrics.json").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    main(parser.parse_args().out.resolve())
