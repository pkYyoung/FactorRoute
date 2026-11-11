"""Compare the frozen capped-buffer rule with a two-stock sub-industry limit."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import search_two_factors as grid
from compare_strategies import ROOT, dump, load_config, sha


RULE = grid.Rule("RSTR", 70, 10, 30, 5)


def constrained_targets(rows, subindustry, limit=2):
    previous = set()
    result = []
    for i, symbols, sectors in rows:
        retained = [j for j in range(15) if symbols[j] in previous]
        by_sector, by_subindustry = Counter(), Counter()
        chosen = []
        for j in (*retained, *range(len(symbols))):
            code, sector = symbols[j], sectors[j]
            sub = (sector, subindustry[code])
            if code in chosen or by_sector[sector] >= 3 or by_subindustry[sub] >= limit:
                continue
            chosen.append(code)
            by_sector[sector] += 1
            by_subindustry[sub] += 1
            if len(chosen) == 10:
                break
        if len(chosen) != 10:
            raise ValueError(f"Cannot fill ten eligible positions at signal index {i}")
        previous = set(chosen)
        result.append((i + 1, np.array(chosen, dtype=int)))
    return result


def weekly_summary(nav):
    close = nav.resample("W-FRI").last()
    weekly = close.div(close.shift().fillna(1)) - 1
    weekly["excess"] = weekly.strategy - weekly.SPY
    result = {}
    for name, frame in (
        ("all", weekly), ("SPY_down", weekly[weekly.SPY < 0]),
        ("SPY_down_2pct", weekly[weekly.SPY <= -.02]),
        ("SPY_down_3pct", weekly[weekly.SPY <= -.03]),
        ("SPY_up_or_flat", weekly[weekly.SPY >= 0]),
    ):
        result[name] = {
            "weeks": len(frame), "wins": int((frame.excess > 1e-12).sum()),
            "strategy_mean": float(frame.strategy.mean()) if len(frame) else None,
            "SPY_mean": float(frame.SPY.mean()) if len(frame) else None,
            "excess_mean": float(frame.excess.mean()) if len(frame) else None,
        }
    return result, weekly


def main(out):
    if out.exists():
        raise FileExistsError(out)
    dates, codes, close, opening, signals, files = grid.inputs()
    rows = grid.build_rankings(signals)["RSTR", 70]
    spy = codes.index("SPY")
    path = ROOT / load_config()["snapshot"] / "universe.csv"
    mapping = pd.read_csv(path).set_index("yahoo_symbol", verify_integrity=True)["GICS Sub-Industry"]
    eligible_ids = {int(code) for _, syms, _ in rows for code in syms}
    subindustry = {i: mapping.get(codes[i]) for i in eligible_ids}
    missing = [codes[i] for i, sub in subindustry.items() if pd.isna(sub) or not str(sub).strip()]
    if missing:
        raise ValueError(f"Missing classifications: {missing}")
    # With a three-stock sector cap, a three-stock sub-industry cap adds no restriction.
    assert all(np.array_equal(a, b) for (_, a), (_, b) in zip(
        grid.targets(rows, RULE), constrained_targets(rows, subindustry, limit=3), strict=True
    ))
    split = dates.searchsorted(grid.TRAIN_END, side="right") - 1
    last = len(dates) - 1
    periods = {
        "full": ([r for r in rows if r[0] < last], last),
        "review": ([r for r in rows if split < r[0] + 1 <= last], last),
    }
    plans = {"baseline": grid.targets(rows, RULE), "subindustry2": constrained_targets(rows, subindustry)}
    by_day = {i: dict(zip(syms, sects, strict=True)) for i, syms, sects in rows}
    for i, ids in plans["subindustry2"]:
        assert len(set(ids)) == 10
        assert max(Counter(by_day[i - 1][j] for j in ids).values()) <= 3
        assert max(Counter((by_day[i - 1][j], subindustry[j]) for j in ids).values()) <= 2
    out.mkdir(parents=True)
    summary = {}
    for period, (signals_in_period, end) in periods.items():
        summary[period] = {}
        for label, plan in (
            ("baseline", grid.targets(signals_in_period, RULE)),
            ("subindustry2", constrained_targets(signals_in_period, subindustry)),
        ):
            m, nav, rolling = grid.evaluate(plan, close, opening, spy, dates, end, grid.FEE)
            stress = grid.evaluate(plan, close, opening, spy, dates, end, grid.STRESS_FEE)[0]
            w, weekly = weekly_summary(nav)
            summary[period][label] = {**m, "excess_15bps": stress["excess"], "weekly": w}
            nav.to_csv(out / f"{period}_{label}_nav.csv", index_label="date")
            rolling.to_csv(out / f"{period}_{label}_13weeks.csv", index=False)
            weekly.to_csv(out / f"{period}_{label}_weekly.csv", index_label="week_end")
    pd.DataFrame([
        (label, dates[i - 1], codes[j], subindustry[j])
        for label, plan in plans.items() for i, ids in plan for j in ids
    ], columns=["rule", "signal", "code", "subindustry"]).to_csv(out / "picks.csv", index=False)
    changed = sum(set(a) != set(b) for (_, a), (_, b) in zip(*plans.values(), strict=True))
    summary["changed_signal_lists_including_oct30"] = changed
    summary["signal_lists"] = len(rows)
    dump(out / "metrics.json", summary)
    dump(out / "input_sha256.json", {str(p.relative_to(ROOT)): sha(p) for p in
                                     [*files, path, Path(__file__).resolve()]})
    print((out / "metrics.json").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out.resolve())
