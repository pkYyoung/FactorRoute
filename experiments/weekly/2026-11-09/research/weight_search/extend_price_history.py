"""Run four frozen recipes over a longer Yahoo price snapshot."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import search_two_factors as grid
from ..strategy_comparison.run import WEEK, ROOT, dump, sha, standardize
from factorcode.data.factors import BetaFactor, MomentumFactor, ResidualVolatilityFactor
from factorcode.data.yfinance_source import load_snapshot
from factorcode.project_config import DASTD_WINDOW, HBETA_WINDOW, RSTR_WINDOW
from factorcode.strategy.weekly import rank_universe
from .test_subindustry_cap import RULE, weekly_summary


RECIPES = {
    "baseline_70_30": (.7, .3, 0, 0),
    "momentum_80_20": (.8, .2, 0, 0),
    "alpha_40_40_20": (.4, 0, .4, .2),
    "low_beta_63_27_10": (.63, .27, 0, .1),
}


def weekly_raw(market):
    close = market["close"]
    date_to_idx = {d: i for i, d in enumerate(close.index)}
    weeks = pd.Series(close.index).groupby(close.index.to_period("W-FRI")).max().tolist()
    spy_return = market["benchmark"]["Adj Close"].pct_change(fill_method=None)
    beta_w = BetaFactor.W_HBETA
    vol_w = ResidualVolatilityFactor.W_DASTD
    # Eleven consecutive exponentially weighted estimates, lagged eleven sessions.
    mom_w = np.convolve(MomentumFactor.W_RSTR, np.ones(11) / 11)
    min_prices = max(len(mom_w) + 11, DASTD_WINDOW, HBETA_WINDOW) + 1
    by_week = {}
    for code in close:
        prices = close[code].dropna()
        if len(prices) < min_prices:
            continue
        ret = prices.pct_change(fill_method=None).dropna()
        x_all = spy_return.reindex(ret.index).to_numpy()
        y_all = ret.to_numpy()
        log_all = np.log1p(np.maximum(y_all, -.9999))
        price_dates = {d: j for j, d in enumerate(ret.index, start=1)}
        for date in weeks:
            j = price_dates.get(date, 0)
            if j < min_prices - 1:
                continue
            y_vol = y_all[j - DASTD_WINDOW:j]
            mean_vol = np.sum(vol_w * y_vol) / vol_w.sum()
            vol = np.sqrt(np.sum(vol_w * (y_vol - mean_vol)**2) / vol_w.sum())
            x, y = x_all[j - HBETA_WINDOW:j], y_all[j - HBETA_WINDOW:j]
            mx, my = np.sum(beta_w * x) / beta_w.sum(), np.sum(beta_w * y) / beta_w.sum()
            beta = np.sum(beta_w * (x - mx) * (y - my)) / (np.sum(beta_w * (x - mx)**2) + 1e-18)
            mom = mom_w @ log_all[j - 11 - (RSTR_WINDOW + 10):j - 11]
            by_week.setdefault(date_to_idx[date], []).append((date, code, mom, vol, beta, my - beta * mx))
    return pd.DataFrame((r for items in by_week.values() for r in items),
                        columns=["date", "code", "RSTR", "DASTD", "HBETA", "HALPHA"])


def prepare(snapshot, start):
    cfg_path = WEEK.parent / "2026-11-02" / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["analysis_start"] = start
    market = load_snapshot(snapshot, cfg["signal_date"])
    raw = weekly_raw(market)
    ranks = rank_universe(raw, market, cfg)
    close = market["close"]
    spy = market["benchmark"]
    closes = close.assign(SPY=spy["Adj Close"])
    opens = market["open"].assign(SPY=spy.Open * spy["Adj Close"] / spy.Close)
    codes = sorted(closes.columns)
    ids = {c: i for i, c in enumerate(codes)}
    ranks["symbol_id"] = ranks.code.map(ids)
    ranks["sector_id"] = pd.factorize(ranks.sector, sort=True)[0]
    dates = closes.index
    panels = []
    for date, group in ranks.groupby("date", sort=True):
        matrix = np.column_stack((group.z_RSTR, -group.z_DASTD,
                                  standardize(group.HALPHA), -standardize(group.HBETA)))
        if not np.isfinite(matrix).all():
            raise ValueError(f"Missing scores on {date}")
        panels.append((dates.get_loc(date), group.symbol_id.to_numpy(),
                       group.sector_id.to_numpy(), matrix))
    return market, ranks, dates, codes, closes[codes].to_numpy(), opens[codes].to_numpy(), panels


def main(snapshot, out):
    if out.exists():
        raise FileExistsError(out)
    market, ranks, dates, codes, close, opening, panels = prepare(snapshot, "2016-01-01")
    first_entry = dates[panels[0][0] + 1]
    spy = codes.index("SPY")
    periods = {
        "full": (first_entry, dates[-1]), "earlier_2016_2023": (first_entry, "2023-12-31"),
        "recent_2024_2026": ("2024-01-01", dates[-1]), "2020": ("2020-01-01", "2020-12-31"),
        "2022": ("2022-01-01", "2022-12-31"),
    }
    out.mkdir(parents=True)
    summary = {}
    for name, weights in RECIPES.items():
        summary[name] = {"weights_RSTR_DASTD_HALPHA_HBETA": weights}
        ranked = []
        for i, ids, sectors, matrix in panels:
            order = np.lexsort((ids, -(matrix @ np.array(weights))))
            ranked.append((i, ids[order], sectors[order]))
        for period, (begin, finish) in periods.items():
            a, b = dates.searchsorted(begin), dates.searchsorted(finish, side="right") - 1
            schedule = grid.targets([row for row in ranked if a <= row[0] + 1 <= b], RULE)
            result, nav, rolling = grid.evaluate(schedule, close, opening, spy, dates, b, grid.FEE)
            w, weekly = weekly_summary(nav)
            summary[name][period] = {**result, "weekly": w}
            nav.to_csv(out / f"{name}_{period}_nav.csv", index_label="date")
            weekly.to_csv(out / f"{name}_{period}_weekly.csv", index_label="week_end")
            rolling.to_csv(out / f"{name}_{period}_13weeks.csv", index=False)
    ranks.to_csv(out / "weekly_rankings.csv", index=False)
    market["quality"].to_csv(out / "data_quality.csv", index=False)
    dump(out / "metrics.json", {"snapshot": str(snapshot), "eligible": {
        "min": int(ranks.groupby("date").size().min()),
        "max": int(ranks.groupby("date").size().max())},
        "strategies": summary})
    files = [Path(__file__).resolve(), Path(grid.__file__).resolve(),
             WEEK.parent / "2026-11-02" / "config.json", snapshot / "manifest.json",
             snapshot / "universe.csv"]
    dump(out / "input_sha256.json", {str(p.relative_to(ROOT)): sha(p) for p in files})
    for name, x in summary.items():
        print(name)
        for period in periods:
            v = x[period]
            print(period, v["start"], v["end"], "return/SPY/DD",
                  [round(100 * v[k], 2) for k in ("net", "SPY", "drawdown")],
                  "weekly", v["weekly"]["all"]["wins"], v["weekly"]["all"]["weeks"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.snapshot.resolve(), args.out.resolve())
