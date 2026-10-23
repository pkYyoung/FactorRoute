"""IC 与 Group 共享的数据变换纯函数。"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VALID_FREQS = ("week", "month", "quarter")
FREQ_ALIASES = {"w": "week", "m": "month", "q": "quarter"}


def add_period(df: pd.DataFrame, date_col: str = "date", freq: str = "W-FRI") -> pd.DataFrame:
    out = df.copy()
    out["period"] = pd.to_datetime(out[date_col]).dt.to_period(freq)
    return out


def compute_period_avg(
    factors: pd.DataFrame, factor_name: str, freq: str = "W-FRI"
) -> pd.DataFrame:
    if factor_name not in factors.columns:
        return pd.DataFrame(columns=["period", "code", f"{factor_name}_avg"])
    frame = factors[["date", "code", factor_name]].dropna(subset=[factor_name]).copy()
    if frame.empty:
        return pd.DataFrame(columns=["period", "code", f"{factor_name}_avg"])
    frame = add_period(frame, freq=freq)
    return frame.groupby(["period", "code"], as_index=False)[factor_name].mean().rename(
        columns={factor_name: f"{factor_name}_avg"}
    )


def cumret_from_daily(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    aligned = series.reindex(index).fillna(0.0).astype(float)
    return (1.0 + aligned).cumprod() - 1.0


def wide_to_long_sorted(
    wide: pd.DataFrame,
    value_col: str,
    index_name: str = "阶段",
) -> pd.DataFrame:
    frame = wide.copy().rename_axis(index_name).reset_index()
    period_col = frame.columns[0]
    long = frame.melt(id_vars=[period_col], var_name="因子", value_name=value_col)
    return long.sort_values([period_col, value_col], ascending=[True, False]).reset_index(drop=True)


def parse_freq_list(raw: str) -> list[str]:
    if not raw or not str(raw).strip():
        return []
    parts = [p.strip().lower() for p in str(raw).replace(";", ",").split(",") if p.strip()]
    result: list[str] = []
    for part in parts:
        token = FREQ_ALIASES.get(part, part)
        if token not in VALID_FREQS:
            raise ValueError(f"无效粒度 {token!r}，可选: {', '.join(VALID_FREQS)}（别名 w,m,q）")
        if token not in result:
            result.append(token)
    return result


def normalize_export_freqs(spec: str | tuple[str, ...] | list[str]) -> list[str]:
    if isinstance(spec, str):
        return parse_freq_list(spec)
    result: list[str] = []
    for item in spec:
        token = FREQ_ALIASES.get(str(item).strip().lower(), str(item).strip().lower())
        if token not in VALID_FREQS:
            raise ValueError(f"无效粒度 {token!r}，可选: {', '.join(VALID_FREQS)}（别名 w,m,q）")
        if token not in result:
            result.append(token)
    return result


def week_label(ts) -> str:
    year, week, _ = pd.Timestamp(ts).isocalendar()
    return f"{year}-W{week:02d}"


def month_label(ts) -> str:
    value = pd.Timestamp(ts)
    return f"{value.year}-{value.month:02d}"


def quarter_label(ts) -> str:
    value = pd.Timestamp(ts)
    return f"{value.year}Q{(value.month - 1) // 3 + 1}"


FREQ_FUNCS = {"week": week_label, "month": month_label, "quarter": quarter_label}


def mean_ic_by_period(
    ic_df: pd.DataFrame,
    factor_cols: list[str],
    freq: str,
    ic_anchor: str = "start",
) -> pd.DataFrame:
    strings = ic_df.index.astype(str)
    if ic_anchor == "start":
        anchors = pd.to_datetime(strings.str.split("/").str[0])
    elif ic_anchor == "end":
        anchors = pd.to_datetime(strings.str.split("/").str[-1])
    else:
        raise ValueError(f"ic_anchor 须为 'start' 或 'end'，收到 {ic_anchor!r}")
    labels = [FREQ_FUNCS[freq](value) for value in anchors]
    subset = ic_df[factor_cols].copy()
    subset["_period"] = labels
    result = subset.groupby("_period", sort=True)[factor_cols].mean()
    result.index.name = "时间区间"
    return result


def cum_ls_by_period(
    premium: pd.DataFrame, factor_cols: list[str], freq: str
) -> pd.DataFrame:
    labels = pd.Series([FREQ_FUNCS[freq](value) for value in premium.index], index=premium.index)
    subset = premium[factor_cols].copy()
    subset["_period"] = labels.values
    rows = []
    periods = sorted(subset["_period"].unique())
    for period in periods:
        part = subset[subset["_period"] == period].drop(columns=["_period"])
        rows.append({
            factor: (float((1 + part[factor].dropna()).prod() - 1) if len(part[factor].dropna()) else float("nan"))
            for factor in factor_cols
        })
    result = pd.DataFrame(rows, index=periods)
    result.index.name = "时间区间"
    return result


def distinct_line_colors(n: int) -> list[str]:
    """按旧图表固定顺序生成可区分颜色。"""
    t10 = list(plt.cm.tab10.colors)
    t20 = list(plt.cm.tab20.colors)
    t20b = list(plt.cm.tab20b.colors)
    pool = t10 + (
        [t20[i] for i in range(0, 20, 2)]
        + [t20[i] for i in range(1, 20, 2)]
        + [t20b[i] for i in range(0, 20, 2)]
        + [t20b[i] for i in range(1, 20, 2)]
    )
    return [pool[i % len(pool)] for i in range(n)]
