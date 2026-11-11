"""Resumable union of the unrestricted 5-point grid and up-to-three 1-point grid."""

import argparse
import fcntl
import hashlib
from itertools import combinations, islice
import json
from math import comb
from pathlib import Path
import subprocess
from time import monotonic

import numpy as np
import pandas as pd

from . import grid as grid


NAMES = grid.NAMES
TOP = 20
TOTAL = 3_504_105
COARSE = 3_108_105
FINE = 411_057
OVERLAP = 15_057
SORT = ["passes_training_filter", "disjoint_median_excess_spy", "rolling_median_excess_spy", "disjoint_worst_dd", *NAMES]
ASCENDING = [False, False, False, True, *([True] * len(NAMES))]


def expanded_recipes():
    for weights in grid.daily.simplex(20, len(NAMES)):
        yield tuple(5 * weight for weight in weights)
    for size in (2, 3):
        for factors in combinations(range(len(NAMES)), size):
            for cuts in combinations(range(1, 100), size - 1):
                points = (0, *cuts, 100)
                weights = [b - a for a, b in zip(points, points[1:])]
                if all(weight % 5 == 0 for weight in weights):
                    continue
                recipe = [0] * len(NAMES)
                for factor, weight in zip(factors, weights, strict=True):
                    recipe[factor] = weight
                yield tuple(recipe)


def write_atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False)
        handle.flush()
    temporary.replace(path)


def panel_digest(panel):
    digest = hashlib.sha256()
    for data in (panel.dates.asi8, panel.signals.asi8, panel.ids, panel.sectors, panel.counts,
                 panel.entries, panel.ends, panel.factors, panel.marks, panel.overnight):
        digest.update(np.ascontiguousarray(data).tobytes())
    return digest.hexdigest()


def fit(out, panel, run, workers, part_size, limit):
    schedule = grid.windows(panel, "training")
    starts = schedule["index"].to_numpy()
    common = np.ones(len(starts), dtype=bool)
    spy = grid.spy_metrics(panel, starts, .001)
    baseline = run(grid.BASE, starts, .001)
    checks = [spy, baseline, run(grid.BASE, starts, .001, True), run(grid.ALPHA, starts, .001)]
    if not np.isfinite(checks).all():
        raise ValueError("A reference has an unavailable training window")

    # Include every possible ranked security in the entry, marks and following-sale audit.
    for week in range(int(starts[-1]) + grid.WIDTH):
        ids = panel.ids[week, :panel.counts[week]]
        quotes = panel.marks[week, :panel.days[week]][:, ids]
        if not (np.isfinite(quotes).all() and (quotes > 0).all()):
            raise ValueError(f"Common training marks unavailable for {panel.signals[week]:%Y-%m-%d}")
        if week < int(starts[-1]) + grid.WIDTH - 1:
            quotes = panel.overnight[week + 1, ids]
            if not (np.isfinite(quotes).all() and (quotes > 0).all()):
                raise ValueError(f"Common training exit prices unavailable for {panel.signals[week]:%Y-%m-%d}")

    total = TOTAL if limit is None else limit
    n = len(NAMES)
    assert comb(n + 19, n - 1) == COARSE
    assert sum(comb(n, k) * comb(99, k - 1) for k in range(1, 4)) == FINE
    assert sum(comb(n, k) * comb(19, k - 1) for k in range(1, 4)) == OVERLAP
    assert COARSE + FINE - OVERLAP == TOTAL
    signature = {"total": total, "part_size": part_size, "panel_sha256": panel_digest(panel),
                 "sources": {path.name: grid.pit.sha(path) for path in
                             (Path(__file__), Path(grid.__file__), Path(grid.__file__).with_name("window_grid.c"))}}
    config = out / "run_config.json"
    if config.exists():
        if json.loads(config.read_text(encoding="utf-8")) != signature:
            raise ValueError("Checkpoint was created with different data, code or batch size")
    else:
        write_atomic(config, signature)

    batches = out / "batches"
    batches.mkdir(exist_ok=True)
    candidates = islice(expanded_recipes(), total)
    processed = passing = 0
    best = None
    start_time = monotonic()
    for index in range((total + part_size - 1) // part_size):
        recipes = list(islice(candidates, part_size))
        record_path = batches / f"{index:05}.json"
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record["count"] != len(recipes):
                raise ValueError(f"Invalid checkpoint count in {record_path}")
        else:
            values = grid.compute(run, recipes, starts, workers, progress=False)
            if not np.isfinite(values).all():
                raise ValueError(f"Batch {index} returned an invalid window after the common-price audit")
            ranked = grid.summarize_grid(recipes, values, schedule, spy, baseline, common)
            eligible = ranked.loc[(ranked[list(NAMES)] != grid.BASE).any(axis=1)]
            record = {"count": len(recipes), "passing": int(ranked.passes_training_filter.sum()),
                      "leaders": eligible.head(TOP).to_dict(orient="records")}
            write_atomic(record_path, record)
        leaders = pd.DataFrame(record["leaders"])
        best = pd.concat([best, leaders], ignore_index=True).sort_values(SORT, ascending=ASCENDING).head(TOP)
        processed += record["count"]
        passing += record["passing"]
        if index % 8 == 0 or processed == total:
            print(f"Saved {processed:,}/{total:,} recipes ({processed / total:.1%}), {monotonic() - start_time:.1f}s", flush=True)

    if processed != total or best.duplicated(list(NAMES)).any():
        raise ValueError("Invalid final candidate count or duplicate finalist")
    schedule.assign(common=common).to_csv(out / "training_window_calendar.csv", index=False)
    best.insert(0, "training_rank", np.arange(len(best)) + 1)
    returns = [run(row, starts, .001)[:, 0] for row in best[list(NAMES)].itertuples(index=False, name=None)]
    correlations = np.corrcoef(returns)
    best["training_return_correlation_at_least_099_with_ranks"] = [
        ",".join(str(j + 1) for j in range(i) if correlations[i, j] >= .99) for i in range(len(best))]
    shortlist = out / "frozen_top20.csv"
    best.to_csv(shortlist, index=False)
    freeze = {"processed": processed, "passing_training_filter": passing, "planned_coarse": COARSE,
              "planned_fine": FINE, "planned_overlap": OVERLAP, "shortlist_sha256": grid.pit.sha(shortlist),
              "training_windows_sha256": grid.pit.sha(out / "training_window_calendar.csv")}
    write_atomic(out / "freeze.json", freeze)
    return common


def main(out, workers, part_size, limit):
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binary = out / "window_grid.dylib"
        native = Path(grid.__file__).with_name("window_grid.c")
        subprocess.run(["rtk", "proxy", "clang", "-O3", "-std=c11", "-ffp-contract=off", "-dynamiclib",
                        str(native), "-o", str(binary)], check=True)
        source = grid.pit.INPUT.parent / "free-data-audit"
        panel = grid.load_panel(source, out)
        run = grid.native_engine(panel, binary)
        grid.validate(panel, run)
        print("Verified the simulator and common nine-factor panel.", flush=True)
        common = fit(out, panel, run, workers, part_size, limit)
        grid.review(out, panel, run, common, "frozen_top20.csv")
        print("Frozen shortlist and historical review saved.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--part-size", type=int, default=8192)
    parser.add_argument("--limit", type=int, help="Smoke-run limit in a separate directory")
    args = parser.parse_args()
    main(args.out, args.workers, args.part_size, args.limit)
