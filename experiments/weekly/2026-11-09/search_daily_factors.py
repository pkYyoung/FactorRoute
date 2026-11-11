"""Search a finite daily-price factor grid under fixed portfolio construction."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import search_two_factors as grid
from compare_strategies import ROOT, dump, relative_peers, sha, standardize
from factorcode.data.factors import BetaFactor
from factorcode.project_config import HBETA_WINDOW
from test_downside_beta import downside_beta
from test_subindustry_cap import RULE, weekly_summary


# All weights are positive on the direction indicated here.
NAMES = ("RSTR", "DASTD_low", "HALPHA", "HSIGMA_low", "HBETA_low",
         "CMRA_low", "peer21", "momentum63", "downside_beta_low")
BASE = (70, 30, 0, 0, 0, 0, 0, 0, 0)
G2 = (0, 30, 0, 0, 0, 0, 40, 30, 0)
DD_LIMIT = -.25


def simplex(units, dimensions):
    if dimensions == 1:
        yield (units,)
        return
    for i in range(units + 1):
        for tail in simplex(units - i, dimensions - 1):
            yield (i, *tail)


def weights_grid():
    result = {tuple(20 * x for x in row) for row in simplex(5, len(NAMES))}
    result |= {BASE, G2}
    for a, b in combinations(range(len(NAMES)), 2):
        for w in range(10, 100, 10):
            v = [0] * len(NAMES)
            v[a], v[b] = w, 100 - w
            result.add(tuple(v))
    for a in range(2, len(NAMES)):
        for w in (10, 20, 30):
            v = [int(.7 * (100 - w) + .5), int(.3 * (100 - w) + .5)] + [0] * (len(NAMES) - 2)
            v[a] = w
            result.add(tuple(v))
    assert all(sum(w) == 100 for w in result)
    return sorted(result)


def factor_panels(dates, close, spy, signals):
    returns = close[1:] / close[:-1] - 1
    months = pd.Series(np.arange(len(dates)), index=dates).groupby(dates.to_period("M")).last().to_numpy()
    down_rows = [(i, s.symbol_id.to_numpy(), s.sector_id.to_numpy()) for i, s in signals]
    down = downside_beta(close, spy, down_rows)
    w = BetaFactor.W_HBETA
    out = []
    for i, s in signals:
        ids = s.symbol_id.to_numpy()
        # Weekly eligibility is fixed across all experiments. Fail on missing inputs.
        block = returns[i - HBETA_WINDOW:i]
        x, y = block[:, spy], block[:, ids]
        if len(x) != HBETA_WINDOW or not np.isfinite(y).all() or not np.isfinite(x).all():
            raise ValueError(f"Incomplete regression input at {dates[i]}")
        mx, my = np.sum(w * x) / w.sum(), (w @ y) / w.sum()
        centered = x - mx
        beta = (w * centered) @ (y - my) / (np.sum(w * centered**2) + 1e-18)
        np.testing.assert_allclose(beta, s.HBETA, rtol=0, atol=1e-10)
        alpha = my - beta * mx
        residual = y - alpha - x[:, None] * beta
        sigma = np.sqrt((w @ (residual - (w @ residual) / w.sum())**2) / w.sum())
        completed = months[months <= i][-13:]
        monthly_close = close[completed][:, ids]
        if len(completed) != 13 or not np.isfinite(monthly_close).all():
            raise ValueError(f"Incomplete month-end prices at {dates[i]}")
        logs = np.diff(np.log(monthly_close), axis=0)
        sums = np.cumsum(logs[::-1], axis=0)
        cmra = sums.max(axis=0) - sums.min(axis=0)
        recent = pd.Series(close[i, ids] / close[i - 21, ids] - 1, index=s.code)
        relative = relative_peers(recent, s.set_index("code").sector)
        medium = close[i, ids] / close[i - 63, ids] - 1

        def z(a):
            return standardize(pd.Series(a)).to_numpy()

        factors = np.column_stack((s.z_RSTR, -s.z_DASTD, z(alpha), -z(sigma),
                                   -z(beta), -z(cmra), z(relative), z(medium), -z(down[i])))
        if not np.isfinite(factors).all():
            raise ValueError(f"Non-finite normalized scores at {dates[i]}")
        out.append((i, ids, s.sector_id.to_numpy(), factors))
    return out


def ranks(panels, weights):
    fraction = np.array(weights) / 100
    rows = []
    for i, ids, sectors, values in panels:
        score = values @ fraction
        order = np.lexsort((ids, -score))
        rows.append((i, ids[order], sectors[order]))
    return rows


def key(metrics, weights):
    return (metrics["rolling_median"], metrics["excess"],
            -metrics["mean_traded_fraction"], -sum(w > 0 for w in weights))


def main(out):
    if out.exists():
        raise FileExistsError(out)
    dates, codes, close, opening, signals, files = grid.inputs()
    spy = codes.index("SPY")
    panels = factor_panels(dates, close, spy, signals)
    split = dates.searchsorted(grid.TRAIN_END, side="right") - 1
    end = len(dates) - 1
    train_panel = [p for p in panels if p[0] + 1 <= split]
    review_panel = [p for p in panels if split < p[0] + 1 <= end]
    full_panel = [p for p in panels if p[0] < end]
    options = weights_grid()
    train = {}

    def run(panel, w, last, cost=grid.FEE):
        return grid.evaluate(grid.targets(ranks(panel, w), RULE), close, opening,
                             spy, dates, last, cost)

    for w in options:
        train[w] = run(train_panel, w, split)[0]
    allowed = [w for w in options if train[w]["drawdown"] >= DD_LIMIT]
    if not allowed:
        raise ValueError("No strategy met the training drawdown screen")
    winner = max(allowed, key=lambda w: key(train[w], w))
    novel = max((w for w in allowed if any(w[2:])), key=lambda w: key(train[w], w))
    # Do not choose a candidate based on these review results.
    review = {w: run(review_panel, w, end)[0] for w in options}
    ref = ROOT / "outputs/weekly/2026-11-09/two-factor-grid/capped_full_nav.csv"
    old = pd.read_csv(ref, index_col=0, parse_dates=True)
    base_nav = run(full_panel, BASE, end)[1]
    pd.testing.assert_index_equal(base_nav.index, old.index, check_names=False)
    np.testing.assert_allclose(base_nav.to_numpy(), old.to_numpy(), rtol=0, atol=1e-10)
    files.append(ref)

    out.mkdir(parents=True)
    pd.DataFrame([
        {**dict(zip(NAMES, w, strict=True)), "passes_train_dd": w in allowed,
         "chosen": w == winner, **{f"train_{k}": v for k, v in train[w].items()},
         **{f"review_{k}": v for k, v in review[w].items()}}
        for w in options
    ]).to_csv(out / "grid.csv", index=False)
    # Reference pair isolates changing only the definition of the low-volatility leg.
    residual_swap = (70, 0, 0, 30, 0, 0, 0, 0, 0)
    summary = {}
    for name, weights in (("baseline", BASE), ("selected", winner), ("selected_novel", novel),
                          ("short_momentum", G2), ("HSIGMA_swap", residual_swap)):
        summary[name] = {"weights": {k: w for k, w in zip(NAMES, weights, strict=True) if w}}
        for period, panel, last in (("train", train_panel, split),
                                    ("review", review_panel, end), ("full", full_panel, end)):
            metrics, nav, rolling = run(panel, weights, last)
            stress = run(panel, weights, last, grid.STRESS_FEE)[0]
            weekly, detail = weekly_summary(nav)
            summary[name][period] = {**metrics, "weekly": weekly, "excess_15bps": stress["excess"]}
            nav.to_csv(out / f"{name}_{period}_nav.csv", index_label="date")
            detail.to_csv(out / f"{name}_{period}_weekly.csv", index_label="week_end")
            rolling.to_csv(out / f"{name}_{period}_13weeks.csv", index=False)
        if name in ("selected", "selected_novel"):
            schedule = grid.targets(ranks(panels, weights), RULE)
            pd.DataFrame([(dates[i - 1], codes[j]) for i, ids in schedule for j in ids],
                         columns=["signal", "code"]).to_csv(out / f"{name}_picks.csv", index=False)
    dump(out / "metrics.json", {
        "grid_size": len(options), "train_eligible": len(allowed), "drawdown_screen": DD_LIMIT,
        "review_winners": sum(v["excess"] > 0 for v in review.values()),
        "selection": "training median 13-week net excess, tie: net excess, lower turnover, fewer descriptors",
        "strategies": summary,
    })
    deps = [Path(__file__).resolve(), Path(grid.__file__).resolve(),
            Path(__file__).with_name("test_downside_beta.py").resolve(),
            Path(__file__).with_name("test_subindustry_cap.py").resolve(),
            ROOT / "factorcode/data/factors.py", ROOT / "factorcode/project_config.py"]
    dump(out / "input_sha256.json", {str(p.relative_to(ROOT)): sha(p) for p in [*files, *deps]})
    print("Rules:", len(options), "Passed training drawdown:", len(allowed))
    for name, s in summary.items():
        print(name, s["weights"], "train median", s["train"]["rolling_median"],
              "review return", s["review"]["net"], "review DD", s["review"]["drawdown"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out.resolve())
