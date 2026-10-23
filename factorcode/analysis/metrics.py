"""IC 与分组回测绩效指标纯函数。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TRADING_DAYS_PER_YEAR = 252


def ic_single_period(factor_values, next_returns, method: str = "spearman") -> float:
    frame = pd.concat(
        [pd.Series(factor_values, name="factor"), pd.Series(next_returns, name="ret")],
        axis=1,
    ).dropna()
    if len(frame) < 3:
        return float("nan")
    if method == "spearman":
        return float(spearmanr(frame["factor"], frame["ret"])[0])
    return float(frame["factor"].corr(frame["ret"], method="pearson"))


def ic_stats(ic_series: pd.Series) -> dict:
    valid = ic_series.dropna()
    if len(valid) == 0:
        return {"ic_mean": np.nan, "ic_std": np.nan, "ic_ir": np.nan,
                "ic_pos_pct": np.nan, "ic_abs_mean": np.nan, "n_periods": 0}
    mean, std = valid.mean(), valid.std()
    return {"ic_mean": mean, "ic_std": std,
            "ic_ir": mean / std if std > 0 else np.nan,
            "ic_pos_pct": (valid > 0).mean(), "ic_abs_mean": valid.abs().mean(),
            "n_periods": len(valid)}


def cumulative_return(daily_returns: pd.Series) -> float:
    valid = daily_returns.dropna()
    return float((1 + valid).prod() - 1) if len(valid) else float("nan")


def annualized_return(cumret: float, n_days: int) -> float:
    return float((1 + cumret) ** (TRADING_DAYS_PER_YEAR / n_days) - 1) if n_days > 0 else float("nan")


def annualized_volatility(daily_returns: pd.Series) -> float:
    valid = daily_returns.dropna()
    std = valid.std()
    return float(std * np.sqrt(TRADING_DAYS_PER_YEAR)) if std > 0 else float("nan")


def sharpe_ratio(ann_ret: float, ann_vol: float) -> float:
    return float(ann_ret / ann_vol) if ann_vol and not np.isnan(ann_vol) else float("nan")


def max_drawdown(daily_returns: pd.Series) -> float:
    valid = daily_returns.dropna()
    if len(valid) == 0:
        return float("nan")
    wealth = (1 + valid).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def calmar_ratio(ann_ret: float, mdd: float) -> float:
    if pd.isna(mdd) or mdd == 0:
        return float("nan")
    return float(ann_ret / abs(mdd))


def win_rate(daily_returns: pd.Series) -> float:
    valid = daily_returns.dropna()
    return float((valid > 0).mean()) if len(valid) else float("nan")


def backtest_stats(premium: pd.Series) -> dict:
    valid = premium.dropna()
    if len(valid) == 0:
        return {"cumret": np.nan, "ann_ret": np.nan, "sharpe": np.nan,
                "vol": np.nan, "max_drawdown": np.nan, "calmar": np.nan,
                "win_rate": np.nan, "n_days": 0}
    cumret = cumulative_return(valid)
    ann_ret = annualized_return(cumret, len(valid))
    vol = annualized_volatility(valid)
    mdd = max_drawdown(valid)
    return {"cumret": cumret, "ann_ret": ann_ret,
            "sharpe": sharpe_ratio(ann_ret, vol), "vol": vol,
            "max_drawdown": mdd, "calmar": calmar_ratio(ann_ret, mdd),
            "win_rate": win_rate(valid), "n_days": len(valid)}
