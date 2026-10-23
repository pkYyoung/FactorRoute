"""分组回测、绩效、切片、阶段分析与全部 Group 绘图。"""
from __future__ import annotations

import argparse
import colorsys
import json
import re
import sys
from pathlib import Path
from typing import Any, List

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from .. import project_config as default_config
from ..project_config import (
    BACKTEST_BENCHMARK_INDEX_COLUMN, FACTOR_COLUMNS, MERGED_COLUMN_TO_FACTOR_CODE,
    PERIOD_FREQ, TOP_PCT, BOTTOM_PCT, WEIGHT_METHOD, index_benchmark_chart_label,
)
from ..data.io import DataIO
from ..data.pipeline import FactorPipeline
from .common import compute_period_avg, cum_ls_by_period, cumret_from_daily, distinct_line_colors, normalize_export_freqs, wide_to_long_sorted
from .metrics import backtest_stats

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
TITLE_FONTSIZE = 16

def load_benchmark_series(
    prem_index: pd.DatetimeIndex,
    benchmark_index: str | None,
) -> tuple[pd.Series | None, str | None]:
    """读取并对齐回测附图使用的基准日收益序列。"""

    col = (benchmark_index or "").strip() or None
    if col is None:
        col = BACKTEST_BENCHMARK_INDEX_COLUMN

    try:

        series = DataIO(default_config.WIDE_CSV_DIR, default_config).load_benchmark_returns(index_column=col)
        print(f"基准来源: {default_config.WIDE_CSV_DIR / 'benchmark_index_daily.csv'} 列「{col}」")
    except Exception as exc:
        print(f"[警告] 未能读取基准指数（请检查 CSV 与列名）: {exc}")
        return None, None

    try:
        return series.reindex(prem_index), index_benchmark_chart_label(col)
    except Exception as exc:
        print(f"[警告] 基准对齐失败: {exc}")
        return None, None


def parse_factors(raw: str | None, available: list[str]) -> list[str]:
    """解析逗号或分号分隔的因子子集。"""
    if not raw or not str(raw).strip():
        return list(available)
    parts = [
        part.strip()
        for part in str(raw).replace(";", ",").split(",")
        if part.strip()
    ]
    bad = [part for part in parts if part not in available]
    if bad:
        raise ValueError(f"未知因子: {bad}，可选: {available}")
    return parts


def plot_top_bottom_panels(
    top_curves: list[tuple[str, pd.Series | None]],
    bottom_curves: list[tuple[str, pd.Series | None]],
    benchmark_curve: pd.Series | None,
    benchmark_label: str,
    title: str,
    ylabel: str,
    fig_path: Path,
    figsize: tuple[float, float] | None = None,
) -> None:
    """按既有 2×N 版式渲染调用方已经准备好的 Top/Bottom 曲线。"""
    n_top = len(top_curves)
    n_bottom = len(bottom_curves)
    n_cols = max(n_top, n_bottom, 1)
    width = max(4.0 * n_cols, 10.0)
    height = 6.8
    if figsize is None:
        figsize = (width, height)

    if n_cols == 1:
        fig, ax_col = plt.subplots(2, 1, figsize=figsize, sharex=True, sharey=True)
        axes = np.array([[ax_col[0]], [ax_col[1]]])
    else:
        fig, axes = plt.subplots(
            2, n_cols, figsize=figsize, sharex=True, sharey=True
        )

    top_count = max(n_top, 1)
    bottom_count = max(n_bottom, 1)
    colors_top = plt.cm.Greens(np.linspace(0.45, 0.9, top_count))
    colors_bottom = plt.cm.Reds(np.linspace(0.45, 0.9, bottom_count))
    fig.suptitle(title, fontsize=12, y=1.02)

    def _plot_one_panel(
        ax,
        curve: pd.Series | None,
        factor: str,
        rank_label: str,
        color,
        linestyle: str,
    ) -> None:
        if curve is None:
            ax.set_title(rank_label, fontsize=10, pad=6)
            ax.text(
                0.5,
                0.5,
                "无数据",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            return
        ax.plot(
            curve.index,
            curve.values,
            color=color,
            linewidth=1.8,
            label=f"{factor}",
            linestyle=linestyle,
            alpha=0.95,
        )
        if benchmark_curve is not None:
            ax.plot(
                benchmark_curve.index,
                benchmark_curve.values,
                color="#1f77b4",
                linewidth=2.0,
                label=f"{benchmark_label}（基准）",
                zorder=10,
            )
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(rank_label, fontsize=10, pad=6)
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)

    for index in range(n_cols):
        ax = axes[0, index]
        if index < n_top:
            factor, curve = top_curves[index]
            _plot_one_panel(
                ax,
                curve,
                factor,
                f"Top {index + 1}：{factor}",
                colors_top[index % top_count],
                "-",
            )
        else:
            ax.axis("off")

    for index in range(n_cols):
        ax = axes[1, index]
        if index < n_bottom:
            factor, curve = bottom_curves[index]
            _plot_one_panel(
                ax,
                curve,
                factor,
                f"Bottom {index + 1}：{factor}",
                colors_bottom[index % bottom_count],
                "--",
            )
        else:
            ax.axis("off")

    middle = min(n_cols // 2, n_cols - 1)
    axes[1, middle].set_xlabel("日期", labelpad=6)
    axes[0, 0].set_ylabel(ylabel, labelpad=8)
    axes[1, 0].set_ylabel(ylabel, labelpad=8)
    for index in range(n_cols):
        plt.setp(axes[1, index].get_xticklabels(), rotation=35, ha="right")
    fig.subplots_adjust(top=0.9, hspace=0.3, wspace=0.18)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    print(f"已保存: {fig_path}")
    plt.close()
def load_correlation_matrix(path: Path) -> pd.DataFrame:
    """宽表方阵：行列为因子名，单元格为 Pearson 相关系数。"""
    df = pd.read_csv(path, index_col=0, encoding='utf-8-sig')
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    m = MERGED_COLUMN_TO_FACTOR_CODE
    df = df.rename(index=lambda x: m.get(x, x), columns=lambda x: m.get(x, x))
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def _abs_corr(corr_df: pd.DataFrame, a: str, b: str) -> float | None:
    if a == b:
        return 1.0
    if a not in corr_df.index or b not in corr_df.columns:
        return None
    try:
        return abs(float(corr_df.loc[a, b]))
    except (KeyError, TypeError, ValueError):
        return None

def _high_corr_with_selected(corr_df: pd.DataFrame | None, name: str, selected: list[str], threshold: float) -> bool:
    if corr_df is None or corr_df.empty:
        return False
    for o in selected:
        v = _abs_corr(corr_df, name, o)
        if v is not None and v >= threshold:
            return True
    return False

def pick_top_bottom_factors(stats_df: pd.DataFrame, n: int, corr_df: pd.DataFrame | None, corr_threshold: float) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    按累计收益排序后贪心选取 Top n / Bottom n；
    若与已选因子 |ρ|≥corr_threshold 则跳过（顺延）。
    返回: (top_names, bottom_names, skipped_top, skipped_bottom)
    """
    if '因子' not in stats_df.columns or '累计收益' not in stats_df.columns:
        raise ValueError('stats 需含列: 因子, 累计收益')
    d = stats_df.copy()
    d['累计收益'] = pd.to_numeric(d['累计收益'], errors='coerce')
    d = d.dropna(subset=['累计收益', '因子'])
    d['因子'] = d['因子'].astype(str).str.strip()
    if len(d) < 2:
        raise ValueError('stats 有效行过少')
    desc = d.sort_values('累计收益', ascending=False)
    asc = d.sort_values('累计收益', ascending=True)
    skipped_top: list[str] = []
    skipped_bottom: list[str] = []
    top: list[str] = []
    for (_, row) in desc.iterrows():
        fac = row['因子']
        if len(top) >= n:
            break
        if _high_corr_with_selected(corr_df, fac, top, corr_threshold):
            skipped_top.append(fac)
            continue
        top.append(fac)
    bottom: list[str] = []
    for (_, row) in asc.iterrows():
        fac = row['因子']
        if len(bottom) >= n:
            break
        if _high_corr_with_selected(corr_df, fac, bottom, corr_threshold):
            skipped_bottom.append(fac)
            continue
        bottom.append(fac)
    if len(top) < n or len(bottom) < n:
        print(f'[警告] 去相关后 Top/Bottom 不足 {n} 个（当前 Top {len(top)}、Bottom {len(bottom)}），可降低 --corr-threshold 或检查相关性表。')
    return (top, bottom, skipped_top, skipped_bottom)

def pick_top_bottom_from_series(factor_returns: pd.Series, n: int, corr_df: pd.DataFrame | None, corr_threshold: float) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    将单阶段「因子 -> 区间累计多空」序列转为 stats 表，再按 pick_top_bottom_factors 贪心选取。
    corr_df 为 None 时等价于按收益取 Top/Bottom 各 n（无顺延）。
    """
    s = factor_returns.dropna().astype(float)
    if len(s) < 2:
        raise ValueError(f'有效因子过少（{len(s)}），无法选取 Top/Bottom {n}')
    stats_df = pd.DataFrame({'因子': s.index.astype(str).str.strip(), '累计收益': s.values})
    return pick_top_bottom_factors(stats_df, n, corr_df, corr_threshold)

def plot_extremes_vs_benchmark(prem: pd.DataFrame, top_names: list[str], bottom_names: list[str], bench_daily: pd.Series | None, bench_display_name: str | None, fig_path: Path, corr_threshold: float | None=None, figsize: tuple[float, float] | None=None) -> None:
    """准备全周期 Top/Bottom 累计曲线并交给共享面板内核渲染。"""
    idx = prem.index.sort_values()
    bench_label = index_benchmark_chart_label(bench_display_name)
    bc = None
    if bench_daily is not None and (not bench_daily.dropna().empty):
        bc = cumret_from_daily(bench_daily, idx)
    sub = f'（|ρ|≥{corr_threshold} 已去相关顺延）' if corr_threshold is not None else ''
    def _prepare_curve(fac: str) -> pd.Series | None:
        if fac not in prem.columns:
            print(f'[跳过] premium 无列: {fac}')
            return None
        return cumret_from_daily(prem[fac], idx)
    top_curves = [(fac, _prepare_curve(fac)) for fac in top_names]
    bottom_curves = [(fac, _prepare_curve(fac)) for fac in bottom_names]
    plot_top_bottom_panels(
        top_curves,
        bottom_curves,
        bc,
        bench_label,
        f'累计收益 Top/Bottom 各因子 vs 基准（子图：单因子 + 基准）{sub}',
        '累计收益率',
        fig_path,
        figsize,
    )

def _apply_x_axis_spacing(ax) -> None:
    """横轴刻度数字与轴线略拉开，避免贴线。"""
    ax.tick_params(axis='x', pad=9)

def compute_cumulative_return(series: pd.Series) -> pd.Series:
    """计算累计收益：(1+r1)*(1+r2)*... - 1"""
    r = series.dropna()
    if r.empty:
        return pd.Series(dtype=float)
    return (1 + r).cumprod() - 1

def plot_curves(df: pd.DataFrame, fig_path: Path=None, figsize: tuple=(12, 6), max_factors: int=8) -> None:
    """
    绘制各因子累计收益曲线。

    df: 日频 Top-Bottom 收益差，index=日期，columns=因子名
    fig_path: 图片保存路径
    max_factors: 单图最多显示的因子数，超过则分多子图
    """
    cols = [c for c in df.columns if df[c].notna().any()]
    if not cols:
        print('无有效因子数据可绘制')
        return
    fig_path = fig_path or FIG_OUT
    n_factors = len(cols)
    colors = distinct_line_colors(n_factors)
    color_by_col = dict(zip(cols, colors))
    if n_factors <= max_factors:
        (fig, ax) = plt.subplots(figsize=figsize)
        for col in cols:
            cum = compute_cumulative_return(df[col])
            if not cum.empty:
                ax.plot(cum.index, cum.values, label=col, color=color_by_col[col], linewidth=1.9, alpha=0.92)
        ax.set_xlabel('日期', labelpad=8)
        ax.set_ylabel('累计收益')
        ax.set_title('因子排序回测：Top-Bottom 组合累计收益', fontsize=TITLE_FONTSIZE, pad=10)
        _apply_x_axis_spacing(ax)
        ax.legend(loc='best', fontsize=9)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
    else:
        n_rows = (n_factors + max_factors - 1) // max_factors
        (fig, axes) = plt.subplots(n_rows, 1, figsize=(figsize[0], figsize[1] * n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]
        for (i, ax) in enumerate(axes):
            start = i * max_factors
            end = min(start + max_factors, n_factors)
            sub_cols = cols[start:end]
            for col in sub_cols:
                cum = compute_cumulative_return(df[col])
                if not cum.empty:
                    ax.plot(cum.index, cum.values, label=col, color=color_by_col[col], linewidth=1.9, alpha=0.92)
            ax.set_ylabel('累计收益')
            ax.legend(loc='best', fontsize=9)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel('日期', labelpad=8)
        _apply_x_axis_spacing(axes[-1])
        fig.suptitle('因子排序回测：Top-Bottom 组合累计收益', fontsize=TITLE_FONTSIZE, y=0.988)
        plt.tight_layout(rect=[0, 0.02, 1, 0.978])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'曲线图已保存: {fig_path}')
    plt.close()

def generate_group_report(
    premium: pd.DataFrame,
    performance_stats: pd.DataFrame,
    output_dir: Path,
    correlation_path: Path,
) -> None:
    """从内存回测结果生成 Group 领域图表。"""
    plot_curves(premium, fig_path=output_dir / 'backtest_curves.png')
    try:
        corr_df = load_correlation_matrix(correlation_path) if correlation_path.exists() else None
        top_names, bottom_names, _, _ = pick_top_bottom_factors(
            performance_stats, 3, corr_df, 0.7
        )
        benchmark, benchmark_name = load_benchmark_series(premium.index, None)
        if benchmark is None or benchmark.dropna().empty:
            benchmark_name = None
        plot_extremes_vs_benchmark(
            premium,
            top_names,
            bottom_names,
            benchmark,
            benchmark_name,
            output_dir / 'plot_extremes_vs_benchmark.png',
            corr_threshold=0.7 if corr_df is not None else None,
        )
    except Exception as exc:
        print(f'\n[提示] 极值 vs 基准图跳过: {exc}')

EDGE_POS = "#d62728"
EDGE_NEG = "#2ca02c"


def _distinct_colors(n: int) -> list:
    t10 = list(plt.cm.tab10.colors)
    t20 = list(plt.cm.tab20.colors)
    pool = t10 + t20
    return [pool[i % len(pool)] for i in range(n)]


def factor_cumret_series(r: pd.Series) -> pd.Series:
    """单日度收益 -> 累计收益率 (1+r).cumprod() - 1；缺失日按 0 收益。"""
    r = r.fillna(0.0)
    return (1.0 + r).cumprod() - 1.0


def bench_cumret_series(daily_ret: pd.Series) -> pd.Series:
    """将基准日收益按缺失日零收益口径转换为累计收益率。"""
    daily_ret = daily_ret.fillna(0.0)
    return (1.0 + daily_ret).cumprod() - 1.0


def subsample_by_freq(cumret_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """按周/月只保留周期末一行（该日快照的累计收益率）。"""
    if cumret_df.empty:
        return cumret_df
    rule = "W-FRI" if freq == "week" else "ME"
    try:
        return cumret_df.resample(rule).last().dropna(how="all")
    except ValueError:
        # 旧版 pandas 月末仍用 M
        return cumret_df.resample("M").last().dropna(how="all")


def plot_stacked_abs_cumret_benchmark(
    prem: pd.DataFrame,
    factor_cols: list[str],
    freq: str,
    bench_daily: pd.Series | None,
    fig_path: Path,
    bench_display_name: str | None = None,
    figsize: tuple[float, float] = (14.0, 6.5),
) -> None:
    """绘制因子累计收益正负堆叠柱和基准累计收益折线。"""
    if freq not in ("week", "month"):
        raise ValueError("freq 须为 week 或 month")

    idx = prem.index.sort_values()
    prem = prem.reindex(idx)

    cumret = pd.DataFrame(
        {fac: factor_cumret_series(prem[fac]) for fac in factor_cols},
        index=idx,
    )
    cumret = subsample_by_freq(cumret, freq)

    if cumret.empty:
        print("无有效数据可绘制")
        return

    dates = cumret.index
    x = mdates.date2num(pd.to_datetime(dates))
    if len(x) > 1:
        bar_w = float(np.min(np.diff(x))) * 0.85
    else:
        bar_w = 5.0

    fig, ax1 = plt.subplots(figsize=figsize)
    colors = _distinct_colors(len(factor_cols))
    bottom_pos = np.zeros(len(cumret))
    bottom_neg = np.zeros(len(cumret))

    for i, fac in enumerate(factor_cols):
        cr = cumret[fac].values.astype(float)
        h_pos = np.where(cr > 0.0, cr, 0.0)
        h_neg = np.where(cr < 0.0, cr, 0.0)
        ec_pos = np.where(h_pos > 0, EDGE_POS, "none")
        ec_neg = np.where(h_neg < 0, EDGE_NEG, "none")
        # 正值：自 0 轴向上堆叠；负值：自 0 轴向下堆叠（h_neg 已为负）
        ax1.bar(
            x,
            h_pos,
            width=bar_w,
            bottom=bottom_pos,
            color=colors[i],
            align="center",
            alpha=0.88,
            edgecolor=ec_pos,
            linewidth=1.1,
        )
        ax1.bar(
            x,
            h_neg,
            width=bar_w,
            bottom=bottom_neg,
            color=colors[i],
            align="center",
            alpha=0.88,
            edgecolor=ec_neg,
            linewidth=1.1,
        )
        bottom_pos = bottom_pos + h_pos
        bottom_neg = bottom_neg + h_neg

    ax1.set_ylabel("累计收益率（正值向上堆叠、负值向下堆叠）；边框红正绿负")
    ax1.set_xlabel("日期")
    ax1.grid(True, axis="y", alpha=0.35)
    ax1.axhline(0, color="gray", linestyle="-", linewidth=0.6, alpha=0.4)

    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    lines2, lab2 = [], []
    bench_label = index_benchmark_chart_label(bench_display_name)
    bench_line_label = f"{bench_label}累计收益率"
    if bench_daily is not None and not bench_daily.dropna().empty:
        b_aligned = bench_daily.reindex(idx).fillna(0.0)
        br_full = bench_cumret_series(b_aligned)
        br_df = subsample_by_freq(pd.DataFrame({"b": br_full}), freq)
        if not br_df.empty:
            br = br_df["b"].reindex(cumret.index)
            ax2 = ax1.twinx()
            ax2.plot(
                dates,
                br.values,
                color="#1f77b4",
                linewidth=2.0,
                marker="o",
                markersize=3,
                label=bench_line_label,
            )
            ax2.set_ylabel(bench_line_label)
            ax2.grid(False)
            lines2, lab2 = ax2.get_legend_handles_labels()

    # 图例只用填充色区分因子；红/绿边框仅出现在图内柱子上表示涨跌
    factor_legend_handles = [
        Patch(
            facecolor=colors[i],
            edgecolor="#cccccc",
            linewidth=0.6,
            label=fac,
        )
        for i, fac in enumerate(factor_cols)
    ]
    ncol_leg = min(8, max(4, len(factor_cols) + 1))
    # 图例用 figure 坐标放在底部（勿用负 y）；loc=lower_center 使图例整体更靠下
    fig.legend(
        factor_legend_handles + lines2,
        factor_cols + lab2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=ncol_leg,
        fontsize=8,
        frameon=True,
    )

    freq_cn = {"week": "周度", "month": "月度"}[freq]
    if bench_daily is not None and not bench_daily.dropna().empty:
        title_bench = f"与{bench_label}"
    else:
        title_bench = "（无基准）"
    ax1.set_title(
        f"因子多空 累计收益率堆叠（正上负下）{title_bench} — {freq_cn}",
        fontsize=15,
        pad=28,
    )

    # top 压低、pad 加大：大标题与柱顶拉开；bottom 留足以免图例与 x 轴标签重叠
    fig.subplots_adjust(
        left=0.07,
        right=0.93,
        top=0.84,
        bottom=0.28,
    )
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", pad_inches=0.18)
    print(f"已保存: {fig_path}")
    plt.close()


def generate_period_report(
    premium: pd.DataFrame,
    output_dir: Path,
    frequency: str,
) -> None:
    """从内存回测结果生成配置指定的周度或月度周期图。"""
    configured_frequency = frequency
    try:
        frequency = frequency.strip().lower()
        if frequency not in ("week", "month"):
            raise ValueError(
                f"STACKED_BARS_FREQ 须为 'week' 或 'month'，当前: {configured_frequency!r}"
            )
        factors = [column for column in FACTOR_COLUMNS if column in premium.columns]
        if factors:
            benchmark, benchmark_name = load_benchmark_series(premium.index, None)
            plot_stacked_abs_cumret_benchmark(
                premium,
                factors,
                frequency,
                benchmark,
                output_dir / f"plot_period_bars_benchmark_{frequency}.png",
                bench_display_name=benchmark_name,
            )
    except Exception as exc:
        print(
            f"\n[提示] 堆叠柱图（STACKED_BARS_FREQ={configured_frequency!r}）跳过: {exc}"
        )


MAX_RRG_POINTS = 10
TWELVE_MONTHS = pd.DateOffset(months=12)

# 轨迹默认：点划线 + 方形端点（非实线折线）；可用 CLI 切换对比
RRG_TRAJECTORY_LINESTYLE = "-."
RRG_MARKER = "s"
RRG_MARKERSIZE = 4.8
RRG_LINEWIDTH = 1.35
# 轨迹末端箭头（参考图：大而实的箭头）
RRG_ARROW_MUTATION_SCALE = 12
RRG_ARROW_LW = 1.6

# 坐标轴：相对数据半幅的边距比例（原约 8%，略紧）；最小半幅避免极差过小时点挤成一团
RRG_AXIS_PAD_FRAC = 0.20
RRG_AXIS_MIN_HALF = 0.8

# 命令行 --trajectory-linestyle 映射到 matplotlib 线型
TRAJECTORY_LINESTYLE_MAP = {
    "dashdot": "-.",
    "dashed": "--",
    "solid": "-",
}


def _figure_size_for_factors(n: int) -> tuple[float, float]:
    """因子数少时用较小画布（如仅 2 个因子），避免图幅过大；16 个因子接近原默认尺寸。"""
    n = max(1, n)
    w_min, w_max = 6.5, 11.0
    h_min, h_max = 4.5, 9.5
    t = min(1.0, max(0.0, (n - 2) / 14.0))
    return (w_min + t * (w_max - w_min), h_min + t * (h_max - h_min))


def _parse_figsize_arg(raw: str) -> tuple[float, float] | None:
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip().replace("×", "x")
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
    elif "x" in s.lower():
        parts = [p.strip() for p in s.lower().split("x", 1) if p.strip()]
    else:
        parts = s.split()
    if len(parts) != 2:
        raise ValueError("figsize 须为 宽,高（英寸），如 10,8 或 10x8")
    return (float(parts[0]), float(parts[1]))

# 四象限角标（对齐路演参考图：红系 / 绿系）
QUADRANT_LABELS = {
    "tr": ("领涨", "#c62828"),  # 右上
    "br": ("走弱", "#81c784"),  # 右下 浅绿
    "bl": ("领跌", "#1b5e20"),  # 左下 深绿
    "tl": ("走强", "#e57373"),  # 左上 浅红
}

QUADRANT_TINTS = {
    "tr": "#fff5f5",
    "br": "#f1f8e9",
    "bl": "#e8f5e9",
    "tl": "#fff0f0",
}


def _rrg_saturate_rgb(rgb: tuple[float, float, float], s_mult: float = 1.14) -> tuple[float, float, float]:
    """略提饱和度、稳住明度，白底上更易区分相邻线。"""
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s = min(1.0, s * s_mult)
    v = max(0.24, min(0.93, v * 1.03))
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return (
        max(0.0, min(1.0, r2)),
        max(0.0, min(1.0, g2)),
        max(0.0, min(1.0, b2)),
    )


def _rrg_distinct_palette() -> list[tuple[float, float, float]]:
    """
    tab20 共 10 组色相；按偶下标优先再补奇下标，使相邻因子尽量落在不同色相上，避免 tab10 前几条蓝/橙挤在一起。
    长度与 len(FACTOR_COLUMNS) 一致（默认 16）。
    """
    t20 = list(plt.cm.tab20.colors)
    n = len(FACTOR_COLUMNS)
    # 偶 0,2,...,18（10 个）+ 奇 1,3,...,11（6 个）= 16，相邻因子尽量不同色相
    full_order = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11]
    if n <= len(full_order):
        order = full_order[:n]
    else:
        order = full_order + [(j % 20) for j in range(n - len(full_order))]
    return [_rrg_saturate_rgb(mcolors.to_rgb(t20[i % len(t20)])) for i in order]


def _line_colors_for_factors(factor_cols: list[str]) -> list:
    """按因子名在 FACTOR_COLUMNS 中的下标取 RRG 专用高区分色。"""
    palette = _rrg_distinct_palette()
    name_to_i = {n: j for j, n in enumerate(FACTOR_COLUMNS)}
    out: list = []
    for name in factor_cols:
        if name in name_to_i:
            out.append(palette[name_to_i[name] % len(palette)])
        else:
            out.append(palette[len(out) % len(palette)])
    return out


def span_is_at_least_twelve_months(first: pd.Timestamp, last: pd.Timestamp) -> bool:
    """判断闭区间跨度是否达到十二个自然月。"""
    first = pd.Timestamp(first).normalize()
    last = pd.Timestamp(last).normalize()
    return (first + TWELVE_MONTHS) <= last


def choose_resample_rule(
    index: pd.DatetimeIndex,
    force: str | None,
) -> tuple[str, str]:
    """按显式选项或样本跨度选择月末、季末取样规则。"""
    if force == "month":
        return "month", "ME"
    if force == "quarter":
        return "quarter", "QE"
    idx = index.sort_values()
    if len(idx) < 2:
        return "month", "ME"
    if span_is_at_least_twelve_months(idx.min(), idx.max()):
        return "quarter", "QE"
    return "month", "ME"


def align_daily_premium_benchmark(
    prem: pd.DataFrame,
    bench: pd.Series,
    factor_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """按共同日期对齐因子与基准，并将缺失日收益填为零。"""
    idx = prem.index.intersection(bench.index).sort_values()
    prem = prem.reindex(idx)
    for c in factor_cols:
        prem[c] = prem[c].fillna(0.0)
    bench = bench.reindex(idx).fillna(0.0)
    return prem, bench


def wealth_at_ends(
    prem: pd.DataFrame,
    bench: pd.Series,
    factor_cols: list[str],
    rule: str,
    max_points: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """由日收益计算净值，在指定期末取样并保留最近若干点。"""
    idx = prem.index
    rows = {}
    for f in factor_cols:
        r = prem[f].astype(float)
        rows[f] = (1.0 + r).cumprod()
    rows["__bench__"] = (1.0 + bench.astype(float)).cumprod()
    wealth = pd.DataFrame(rows, index=idx)
    try:
        snap = wealth.resample(rule).last()
    except ValueError:
        snap = wealth.resample("M" if rule == "ME" else "Q").last()
    snap = snap.dropna(how="all")
    snap = snap.dropna(subset=["__bench__"])
    if len(snap) > max_points:
        snap = snap.iloc[-max_points:]
    return snap, snap["__bench__"]


def compute_rs_mom(
    snap: pd.DataFrame,
    factor_cols: list[str],
    mom_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    相对强弱 RS = ln(V^f/V^b)。snap 中列为复利净值 V=∏(1+r)，不是「累计收益率」CR=(∏(1+r))−1。

    若误用两个累计收益率之比 CR_f/CR_b，在两边一亏一赚时比值可能为负，取 ln 无实数意义。
    用净值之比时 V^f,V^b>0（通常每日 r>−100% 则 V 恒正），故 V^f/V^b>0，ln 有定义。

    动量 Mom = RS − RS.shift(mom_k)：同一因子在相邻取样点上的 RS 差分；mom_k 为期数（月或季，与出图频率一致）。
    """
    vb = snap["__bench__"].replace(0.0, np.nan)
    rs = pd.DataFrame(index=snap.index)
    mom = pd.DataFrame(index=snap.index)
    for f in factor_cols:
        vf = snap[f].replace(0.0, np.nan)
        rs[f] = np.log(vf / vb)
        mom[f] = rs[f] - rs[f].shift(mom_k)
    return rs, mom


def _collect_xy_bounds(
    rs: pd.DataFrame,
    mom: pd.DataFrame,
    factor_cols: list[str],
    *,
    pad_frac: float | None = None,
    min_half: float | None = None,
) -> tuple[float, float, float, float]:
    """
    以 (0,0) 为中心对称取轴范围：横轴左右、纵轴上下到原点的距离分别相等，便于四象限视觉平衡。
    pad_frac：在数据半幅上的相对边距（默认 RRG_AXIS_PAD_FRAC）。
    min_half：横、纵轴各自的最小半幅；数据极差很小时仍留出空白，避免与参考图相比「全挤在中间」。
    """
    pf = RRG_AXIS_PAD_FRAC if pad_frac is None else float(pad_frac)
    mh = RRG_AXIS_MIN_HALF if min_half is None else float(min_half)
    xs, ys = [], []
    for f in factor_cols:
        m = rs[f].notna() & mom[f].notna()
        if m.any():
            xs.extend(rs[f][m].astype(float).tolist())
            ys.extend(mom[f][m].astype(float).tolist())
    if not xs:
        h = max(mh, 1.0)
        return -h, h, -h, h

    half_x = max(abs(min(xs)), abs(max(xs)))
    half_y = max(abs(min(ys)), abs(max(ys)))
    half_x = max(half_x, 1e-12)
    half_y = max(half_y, 1e-12)
    pad_x = max(half_x * pf, 0.03)
    pad_y = max(half_y * pf, 0.03)
    hx = max(half_x + pad_x, mh)
    hy = max(half_y + pad_y, mh)
    return -hx, hx, -hy, hy


def _add_quadrant_backgrounds(ax: plt.Axes, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    """以 (0,0) 为界分四象限铺淡色底。"""
    if xmax > 0 and ymax > 0:
        ax.add_patch(
            Rectangle(
                (0, 0),
                xmax,
                ymax,
                facecolor=QUADRANT_TINTS["tr"],
                edgecolor="none",
                zorder=0,
                alpha=0.42,
            )
        )
    if xmin < 0 and ymax > 0:
        ax.add_patch(
            Rectangle(
                (xmin, 0),
                -xmin,
                ymax,
                facecolor=QUADRANT_TINTS["tl"],
                edgecolor="none",
                zorder=0,
                alpha=0.42,
            )
        )
    if xmin < 0 and ymin < 0:
        ax.add_patch(
            Rectangle(
                (xmin, ymin),
                -xmin,
                -ymin,
                facecolor=QUADRANT_TINTS["bl"],
                edgecolor="none",
                zorder=0,
                alpha=0.42,
            )
        )
    if xmax > 0 and ymin < 0:
        ax.add_patch(
            Rectangle(
                (0, ymin),
                xmax,
                -ymin,
                facecolor=QUADRANT_TINTS["br"],
                edgecolor="none",
                zorder=0,
                alpha=0.42,
            )
        )


def _nice_tick_step(span: float, target_segments: int = 8) -> float:
    """将轴域均分为约 target_segments 段时的「整齐」步长（如 0.2、0.5）。"""
    if span <= 0 or not np.isfinite(span):
        return 0.1
    raw = span / float(target_segments)
    if raw <= 0:
        return 0.1
    magnitude = 10 ** np.floor(np.log10(raw))
    normalized = raw / magnitude
    if normalized < 1.5:
        step = magnitude
    elif normalized < 3.5:
        step = 2 * magnitude
    elif normalized < 7.5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    return float(step)


def _style_axes_l_spines_ticks(ax: plt.Axes) -> None:
    """
    四边 spine 均关闭，避免出现左/下浅灰「画框」线。
    主刻度与数字已全部画在中心十字线上（_draw_crosshair_ticks），不再依赖边沿 spine。
    """
    for s in ax.spines.values():
        s.set_visible(False)
    # 数字与刻度线由 _draw_crosshair_ticks 画在中心黑十字线上
    ax.tick_params(
        axis="x",
        which="major",
        length=0,
        width=0,
        bottom=True,
        top=False,
        labelbottom=False,
        labeltop=False,
    )
    ax.tick_params(
        axis="y",
        which="major",
        length=0,
        width=0,
        left=True,
        right=False,
        labelleft=False,
        labelright=False,
    )
    ax.tick_params(axis="both", which="minor", left=False, bottom=False)


def _add_quadrant_corner_labels(ax: plt.Axes, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    dx = xmax - xmin
    dy = ymax - ymin
    off_x = max(dx * 0.02, 0.01)
    off_y = max(dy * 0.02, 0.01)
    kw = dict(fontsize=12, fontweight="bold", zorder=3)
    ax.text(xmax - off_x, ymax - off_y, QUADRANT_LABELS["tr"][0], ha="right", va="top", color=QUADRANT_LABELS["tr"][1], **kw)
    ax.text(xmax - off_x, ymin + off_y, QUADRANT_LABELS["br"][0], ha="right", va="bottom", color=QUADRANT_LABELS["br"][1], **kw)
    ax.text(xmin + off_x, ymin + off_y, QUADRANT_LABELS["bl"][0], ha="left", va="bottom", color=QUADRANT_LABELS["bl"][1], **kw)
    ax.text(xmin + off_x, ymax - off_y, QUADRANT_LABELS["tl"][0], ha="left", va="top", color=QUADRANT_LABELS["tl"][1], **kw)


def _setup_grid_and_crosshairs(ax: plt.Axes) -> None:
    """浅灰实线网格；中心十字轴加粗（强于网格）。"""
    ax.set_axisbelow(True)
    ax.grid(
        True,
        linestyle="-",
        linewidth=0.45,
        color="#c8c8c8",
        alpha=0.95,
    )
    ax.axhline(0.0, color="#424242", linestyle="-", linewidth=1.35, zorder=2)
    ax.axvline(0.0, color="#424242", linestyle="-", linewidth=1.35, zorder=2)


def _apply_rrg_tick_locators(ax: plt.Axes) -> None:
    """等间隔主刻度（如 0.2、0.08），与路演图一致且保证横轴刻度线有位可画。"""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    step_x = _nice_tick_step(xmax - xmin)
    step_y = _nice_tick_step(ymax - ymin)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(step_x))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(step_y))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))


def _fmt_tick_value(v: float) -> str:
    return f"{v:g}"


def _draw_crosshair_ticks(ax: plt.Axes) -> None:
    """
    主刻度短线与数字紧贴在中心十字线（y=0、x=0）上。
    横轴：刻度线仅在 y=0 下方；纵轴：仅在 x=0 左侧，且纵刻度线较短、与数字留出间距。
    原点「0」标在横轴（y=0）一侧、靠竖轴左下，不单独占纵轴刻度位。
    """
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    span_x = xmax - xmin
    span_y = ymax - ymin
    # 纵轴刻度线较短，避免与左侧数字相碰
    dx = max(span_x * 0.007, 1e-9)
    dy = max(span_y * 0.014, 1e-9)
    xticks = [float(t) for t in ax.get_xticks()]
    yticks = [float(t) for t in ax.get_yticks()]
    tc = "#212121"
    lw = 1.15
    zl = 4
    zt = 5
    pad_pt_x = 5
    pad_pt_y = 11

    for x in xticks:
        if x < xmin - 1e-12 or x > xmax + 1e-12:
            continue
        ax.plot([x, x], [0.0, -dy], color=tc, lw=lw, zorder=zl, clip_on=False)
        if abs(x) < 1e-10:
            # 「0」标在横轴上、相对原点靠左（勿用纵轴靠下居中样式）
            ax.annotate(
                "0",
                xy=(0.0, 0.0),
                xytext=(-pad_pt_y, -pad_pt_x),
                textcoords="offset points",
                ha="right",
                va="top",
                fontsize=11,
                color=tc,
                zorder=zt,
                annotation_clip=False,
            )
        else:
            ax.annotate(
                _fmt_tick_value(x),
                xy=(x, 0.0),
                xytext=(0, -pad_pt_x),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=11,
                color=tc,
                zorder=zt,
                annotation_clip=False,
            )

    for y in yticks:
        if y < ymin - 1e-12 or y > ymax + 1e-12:
            continue
        ax.plot([0.0, -dx], [y, y], color=tc, lw=lw, zorder=zl, clip_on=False)
        if abs(y) < 1e-12:
            continue
        ax.annotate(
            _fmt_tick_value(y),
            xy=(0.0, y),
            xytext=(-pad_pt_y, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=11,
            color=tc,
            zorder=zt,
            annotation_clip=False,
        )


def _place_axis_labels_axes(ax: plt.Axes) -> None:
    """
    与图中蓝框一致（数据坐标）：
    - 纵轴标题：水平仍在竖线顶端**右侧**；竖直方向与象限角标「走强」同高（与 _add_quadrant_corner_labels 的 tl 一致：y = ymax - off_y）；
    - 横轴标题：在**水平中心轴 y=0** 与右边界交点 (xmax, 0) 下方。
    须在 tight_layout 之后调用，以便 xlim/ylim 已定型。
    """
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    dy = ymax - ymin
    off_y = max(dy * 0.02, 0.01)
    y_same_as_tl = ymax - off_y
    ax.annotate(
        "相对强弱的动量",
        xy=(0.0, y_same_as_tl),
        xytext=(8, 0),
        textcoords="offset points",
        xycoords="data",
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#212121",
        zorder=10,
        annotation_clip=False,
    )
    ax.annotate(
        "相对强弱",
        xy=(xmax, 0.0),
        xytext=(0, -14),
        textcoords="offset points",
        xycoords="data",
        ha="right",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#212121",
        zorder=10,
        annotation_clip=False,
    )


def plot_rrg_presentation(
    rs: pd.DataFrame,
    mom: pd.DataFrame,
    factor_cols: list[str],
    out_path: Path,
    freq_cn: str,
    snap_index: pd.DatetimeIndex,
    figsize: tuple[float, float] | None = None,
    *,
    trajectory_linestyle: str | None = None,
    trajectory_marker: str | None = None,
    markersize: float | None = None,
    axis_pad_frac: float | None = None,
    axis_min_half: float | None = None,
) -> None:
    """按既有四象限样式渲染已计算的 RS 与动量轨迹。"""
    ls = trajectory_linestyle if trajectory_linestyle is not None else RRG_TRAJECTORY_LINESTYLE
    mk = trajectory_marker if trajectory_marker is not None else RRG_MARKER
    ms = markersize if markersize is not None else RRG_MARKERSIZE
    colors = _line_colors_for_factors(factor_cols)
    xmin, xmax, ymin, ymax = _collect_xy_bounds(
        rs,
        mom,
        factor_cols,
        pad_frac=axis_pad_frac,
        min_half=axis_min_half,
    )

    fs = figsize if figsize is not None else _figure_size_for_factors(len(factor_cols))
    fig, ax = plt.subplots(figsize=fs, facecolor="white")
    print(f"画布尺寸（英寸）: {fs[0]:.2f} × {fs[1]:.2f}（因子数 {len(factor_cols)}）")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    _add_quadrant_backgrounds(ax, xmin, xmax, ymin, ymax)
    _add_quadrant_corner_labels(ax, xmin, xmax, ymin, ymax)
    _setup_grid_and_crosshairs(ax)
    _apply_rrg_tick_locators(ax)

    for i, f in enumerate(factor_cols):
        x = rs[f]
        y = mom[f]
        m = x.notna() & y.notna()
        if not m.any():
            print(f"[警告] 因子 {f} 无有效 RS/Mom 点，跳过")
            continue
        xv = x[m].values.astype(float)
        yv = y[m].values.astype(float)
        ax.plot(
            xv,
            yv,
            linestyle=ls,
            marker=mk,
            color=colors[i],
            linewidth=RRG_LINEWIDTH,
            markersize=ms,
            markerfacecolor=colors[i],
            markeredgecolor="white",
            markeredgewidth=0.7,
            alpha=0.92,
            zorder=3,
            label=f,
            dash_capstyle="round",
        )
        if len(xv) >= 2:
            ax.annotate(
                "",
                xy=(xv[-1], yv[-1]),
                xytext=(xv[-2], yv[-2]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=colors[i],
                    lw=RRG_ARROW_LW,
                    mutation_scale=RRG_ARROW_MUTATION_SCALE,
                    shrinkA=2,
                    shrinkB=2,
                ),
                zorder=5,
            )

    _style_axes_l_spines_ticks(ax)
    _draw_crosshair_ticks(ax)

    as_of = snap_index.max()
    as_of_str = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    # 标题：量化风格相对轮动图：截止日：季度/月度（随样本期末与出图频率变化）
    # va=top + y 与 tight_layout(rect) 的 top 配合：子图顶约 0.91，标题底缘贴近其上方，避免大块留白
    fig.text(
        0.5,
        0.965,
        f"量化风格相对轮动图：{as_of_str}：{freq_cn}",
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
        color="#1a1a1a",
        transform=fig.transFigure,
    )

    fig.text(0.98, 0.018, "数据来源：FactorRoute 因子研究（因子多空溢价）", ha="right", fontsize=8, color="#888888")

    ncol = 8 if len(factor_cols) >= 12 else 4
    # bbox 为 axes 坐标：负值越小图例离图越远；略收紧以靠近 x 轴下方
    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=ncol,
        fontsize=7,
        frameon=False,
        columnspacing=0.8,
        handlelength=2.2,
    )
    leg.set_zorder(6)

    # [left,bottom,right,top]：子图区略放大；top 不宜过大，否则与 va=top 的标题重叠
    plt.tight_layout(rect=[0.06, 0.16, 0.94, 0.91])
    _place_axis_labels_axes(ax)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    print(f"已保存: {out_path}")
    plt.close()


def generate_rrg_report(
    premium: pd.DataFrame,
    out_path: Path,
    *,
    factor_names: str | None = None,
    benchmark_index: str | None = None,
    force_freq: str | None = None,
    mom_k: int = 1,
    max_points: int = MAX_RRG_POINTS,
    figsize: tuple[float, float] | str | None = None,
    trajectory_linestyle: str = RRG_TRAJECTORY_LINESTYLE,
    trajectory_marker: str = RRG_MARKER,
    axis_pad_frac: float | None = None,
    axis_min_half: float | None = None,
    verbose: bool = False,
) -> bool:
    """完成 RRG 的因子选择、数据准备、计算与绘图。"""
    factor_cols = [column for column in FACTOR_COLUMNS if column in premium.columns]
    if not factor_cols:
        print("premium 表中无已知因子列" if verbose else "\n[附图] RRG：无可用因子列，跳过")
        return False
    factor_cols = parse_factors(factor_names, factor_cols)

    benchmark, _ = load_benchmark_series(premium.index, benchmark_index)
    if benchmark is None or benchmark.dropna().empty:
        print("[错误] 无基准日收益，无法计算 RRG。" if verbose else "\n[附图] RRG：无基准日收益，跳过")
        return False

    premium_rrg, benchmark_rrg = align_daily_premium_benchmark(
        premium, benchmark, factor_cols
    )
    frequency, rule = choose_resample_rule(premium_rrg.index, force_freq)
    if verbose:
        print(
            f"样本: {premium_rrg.index.min().date()} ~ {premium_rrg.index.max().date()} | "
            f"出图频率: {frequency}（rule={rule}）| 因子数: {len(factor_cols)}"
        )

    snapshots, _ = wealth_at_ends(
        premium_rrg,
        benchmark_rrg,
        factor_cols,
        rule,
        max_points=max_points,
    )
    if snapshots.empty or len(snapshots) < 2:
        print(
            "期末取样过少，无法绘制 RRG（至少需要 2 个期末点）。"
            if verbose
            else "[附图] RRG：期末点不足 2 个，跳过"
        )
        return False

    relative_strength, momentum = compute_rs_mom(
        snapshots, factor_cols, max(1, mom_k)
    )
    frequency_cn = "季度" if frequency == "quarter" else "月度"
    if isinstance(figsize, str):
        figsize = _parse_figsize_arg(figsize)
    if verbose:
        linestyle_name = next(
            (
                name
                for name, value in TRAJECTORY_LINESTYLE_MAP.items()
                if value == trajectory_linestyle
            ),
            trajectory_linestyle,
        )
        pad = RRG_AXIS_PAD_FRAC if axis_pad_frac is None else axis_pad_frac
        min_half = RRG_AXIS_MIN_HALF if axis_min_half is None else axis_min_half
        print(
            f"轨迹样式: {linestyle_name} → {trajectory_linestyle!r}，标记: {trajectory_marker!r}"
        )
        print(f"坐标轴留白: 相对边距={pad}，最小半幅={min_half}")

    plot_rrg_presentation(
        relative_strength,
        momentum,
        factor_cols,
        out_path,
        freq_cn=frequency_cn,
        snap_index=snapshots.index,
        figsize=figsize,
        trajectory_linestyle=trajectory_linestyle,
        trajectory_marker=trajectory_marker,
        axis_pad_frac=axis_pad_frac,
        axis_min_half=axis_min_half,
    )
    return True


def _factor_cols(prem: pd.DataFrame) -> list[str]:
    return [c for c in FACTOR_COLUMNS if c in prem.columns]


def cumret_within_phase(r: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """阶段内、从阶段首交易日起的累计收益率。"""
    mask = (r.index >= start) & (r.index <= end)
    sub = r.loc[mask].sort_index().fillna(0.0).astype(float)
    if sub.empty:
        return pd.Series(dtype=float)
    return (1.0 + sub).cumprod() - 1.0


def plot_one_phase_extremes_vs_benchmark(
    prem: pd.DataFrame,
    phase: dict,
    top_names: list[str],
    bottom_names: list[str],
    bench_daily: pd.Series | None,
    bench_display_name: str | None,
    fig_path: Path,
    figsize: tuple[float, float] | None = None,
    corr_threshold: float | None = None,
) -> None:
    """准备单阶段内从首日起累计的极值曲线并渲染共享面板。"""

    start = pd.Timestamp(phase["start"])
    end = pd.Timestamp(phase["end"])
    phase_label = str(phase.get("label", f"{start.date()}~{end.date()}")).strip()
    bench_label = index_benchmark_chart_label(bench_display_name)

    window_idx = prem.index[(prem.index >= start) & (prem.index <= end)].sort_values()
    if len(window_idx) < 2:
        print(f"[跳过] 阶段「{phase_label}」有效交易日不足: {len(window_idx)}")
        return

    bc = None
    if bench_daily is not None and not bench_daily.dropna().empty:
        bc = cumret_within_phase(bench_daily, start, end)
    dedup_suffix = (
        f"（|ρ|≥{corr_threshold} 已去相关顺延）"
        if corr_threshold is not None
        else ""
    )
    def _prepare_curve(fac: str) -> pd.Series | None:
        if fac not in prem.columns:
            print(f"[跳过] premium 无列: {fac}")
            return None
        c = cumret_within_phase(prem[fac], start, end)
        if c.empty:
            return None
        return c
    top_curves = [(fac, _prepare_curve(fac)) for fac in top_names]
    bottom_curves = [(fac, _prepare_curve(fac)) for fac in bottom_names]
    plot_top_bottom_panels(
        top_curves,
        bottom_curves,
        bc if bc is not None and not bc.empty else None,
        bench_label,
        f"{phase_label}：区间内 Top/Bottom vs 基准（累计自区间首日起）{dedup_suffix}",
        "累计收益率（区间内）",
        fig_path,
        figsize,
    )


def _safe_filename_fragment(label: str) -> str:
    s = re.sub(r'[<>:"/\\|?*]', "_", label)
    s = re.sub(r"\s+", "_", s.strip())
    return s[:80] if len(s) > 80 else s


def run_all_phases(
    prem_path: Path,
    phases: list[dict],
    out_dir: Path,
    n: int,
    bench_daily: pd.Series | None,
    bench_display_name: str | None,
    corr_df: pd.DataFrame | None = None,
    corr_threshold: float = 0.7,
) -> None:
    """按配置顺序选择各阶段极值因子并逐阶段生成图表。"""
    prem = pd.read_csv(
        prem_path, index_col=0, parse_dates=True, encoding="utf-8-sig"
    )
    if prem.index.tz is not None:
        prem.index = prem.index.tz_localize(None)
    prem = prem.sort_index()

    factor_cols = _factor_cols(prem)
    if not factor_cols:
        raise ValueError("premium 表中无 FACTOR_COLUMNS 对齐列")

    wide = _build_phase_wide(prem, factor_cols, phases)
    use_dedup = corr_df is not None and not corr_df.empty

    out_dir.mkdir(parents=True, exist_ok=True)
    # 与 phases 顺序一一对应，避免 JSON 中重复 label 时 wide.loc[label] 取到多行
    if len(wide) != len(phases):
        raise ValueError(f"阶段数与宽表行数不一致: phases={len(phases)}, wide={len(wide)}")
    for i, ph in enumerate(phases):
        label = str(ph["label"]).strip()
        row = wide.iloc[i].dropna()
        if len(row) < 2:
            print(f"[警告] 阶段「{label}」有效因子不足 2 个，跳过作图")
            continue
        if use_dedup:
            top_names, bottom_names, skipped_top, skipped_bottom = (
                pick_top_bottom_from_series(row, n, corr_df, corr_threshold)
            )
            if skipped_top or skipped_bottom:
                print(
                    f"  阶段「{label}」顺延跳过 Top: {skipped_top}；Bottom: {skipped_bottom}"
                )
        else:
            top_names = row.nlargest(n).index.tolist()
            bottom_names = row.nsmallest(n).index.tolist()

        slug = _safe_filename_fragment(label)
        fig_path = out_dir / f"plot_phases_extremes_{i:02d}_{slug}.png"
        plot_one_phase_extremes_vs_benchmark(
            prem,
            ph,
            top_names,
            bottom_names,
            bench_daily,
            bench_display_name,
            fig_path,
            corr_threshold=corr_threshold if use_dedup else None,
        )


def generate_phase_report(*args, **kwargs) -> None:
    """使用既有阶段参数生成全部 Phase 领域图表。"""
    run_all_phases(*args, **kwargs)


def get_top_bottom_codes(
    factor_avg: pd.DataFrame,
    factor_col: str,
    top_pct: float = TOP_PCT,
    bottom_pct: float = BOTTOM_PCT,
) -> pd.DataFrame:
    """
    按因子值截面排序，取 Top / Bottom 分组。

    输入: factor_avg 列 (period, code, {factor_col})
    输出: 列 (period, code, group)，group in ['top','bottom']
    """
    avg = factor_avg.dropna(subset=[factor_col])
    if avg.empty:
        return pd.DataFrame(columns=["period", "code", "group"])

    out_list = []
    for p, g in avg.groupby("period"):
        g = g.sort_values(factor_col, ascending=False).reset_index(drop=True)
        n = len(g)
        if n == 0:
            continue
        n_top = max(1, int(n * top_pct))
        n_bot = max(1, int(n * bottom_pct))
        g["group"] = ""
        g.loc[g.index[:n_top], "group"] = "top"
        g.loc[g.index[-n_bot:], "group"] = "bottom"
        sub = g[g["group"] != ""][["code", "group"]].copy()
        sub["period"] = p
        out_list.append(sub[["period", "code", "group"]])

    if not out_list:
        return pd.DataFrame(columns=["period", "code", "group"])
    return pd.concat(out_list, ignore_index=True)


def _build_date_prev_map(dates: pd.Series) -> dict:
    """
    构建交易日 -> 前一交易日的映射。

    输入: dates 为日期序列（含重复）
    输出: {Timestamp(d): Timestamp(前一交易日)}，用于市值加权时取前一日
    """
    uniq = sorted(pd.to_datetime(dates).unique())
    return {pd.Timestamp(d): pd.Timestamp(uniq[i - 1]) for i, d in enumerate(uniq) if i > 0}


def compute_daily_premium(
    returns: pd.DataFrame,
    mkt_cap: pd.DataFrame,
    top_bottom: pd.DataFrame,
    period: pd.Period,
    date_prev_map: dict,
    weight_method: str = WEIGHT_METHOD,
) -> pd.Series:
    """
    计算下一周期内，每天 Top 组合与 Bottom 组合的收益差。

    本周期末的组合用于下一周期持有，因此计算的是 period+1 对应日期的收益差。
    市值加权使用前一交易日市值。

    输入:
        returns: 列 (date, code, ret)
        mkt_cap: 列 (date, code, mkt_cap)
        top_bottom: 列 (period, code, group)
        period: 当前周期（如 2025-01-04/2025-01-10 或 2025-01）
        date_prev_map: 交易日 -> 前一交易日
    输出: Series，index=日期，value=当日 Top-Bottom 收益差
    """
    next_start = (period + 1).start_time
    next_end = (period + 1).end_time
    next_dates = pd.date_range(next_start, next_end, freq="B")

    tb = top_bottom[top_bottom["period"] == period]
    if tb.empty:
        return pd.Series(dtype=float)

    top_codes = set(tb[tb["group"] == "top"]["code"])
    bot_codes = set(tb[tb["group"] == "bottom"]["code"])

    prem = []
    for d in next_dates:
        ret_d = returns[returns["date"] == d]
        if ret_d.empty:
            continue

        # 前一交易日市值用于加权
        prev_d = date_prev_map.get(pd.Timestamp(d))
        if prev_d is None:
            continue
        cap_prev = mkt_cap[mkt_cap["date"] == prev_d]
        if cap_prev.empty:
            continue

        def _port_ret(codes: set, r: pd.DataFrame, c_prev: pd.DataFrame) -> float:
            """
            计算指定股票组合当日的加权收益。

            输入:
                codes: 股票代码集合（Top 或 Bottom 组合）
                r: 当日收益，列 (date, code, ret)
                c_prev: 前一交易日市值，列 (date, code, mkt_cap)
            输出: float，组合当日收益（等权或市值加权，由 weight_method 决定）
            """
            r = r[r["code"].isin(codes)]
            c_prev = c_prev[c_prev["code"].isin(codes)]
            merged = r.merge(c_prev[["code", "mkt_cap"]], on="code", how="inner")
            if merged.empty:
                return np.nan
            if weight_method == "等权":
                merged["weight"] = 1.0 / len(merged)
            else:
                tot = merged["mkt_cap"].sum()
                if tot <= 0:
                    return np.nan
                merged["weight"] = merged["mkt_cap"] / tot
            return (merged["ret"] * merged["weight"]).sum()

        top_ret = _port_ret(top_codes, ret_d, cap_prev)
        bot_ret = _port_ret(bot_codes, ret_d, cap_prev)
        if np.isnan(top_ret) or np.isnan(bot_ret):
            continue
        prem.append((d, top_ret - bot_ret))

    if not prem:
        return pd.Series(dtype=float)
    return pd.Series({d: v for d, v in prem})


def run_single_factor_backtest(
    factors: pd.DataFrame,
    returns: pd.DataFrame,
    mkt_cap: pd.DataFrame,
    factor_name: str,
) -> pd.Series:
    """
    对单个因子运行回测，返回日频 Top-Bottom 收益差序列。

    输入:
        factors: 列 (date, code, factor1, factor2, ...)
        returns: 列 (date, code, ret)
        mkt_cap: 列 (date, code, mkt_cap)
        factor_name: 因子列名
    输出: Series，index=日期（从第二周起），value=当日 Top-Bottom 收益差
    """
    factor_avg = compute_period_avg(factors, factor_name, PERIOD_FREQ)
    if factor_avg.empty:
        return pd.Series(dtype=float)

    top_bottom = get_top_bottom_codes(
        factor_avg,
        f"{factor_name}_avg",
        top_pct=TOP_PCT,
        bottom_pct=BOTTOM_PCT,
    )
    if top_bottom.empty:
        return pd.Series(dtype=float)

    periods = sorted(top_bottom["period"].unique())
    all_dates = pd.concat([returns["date"], mkt_cap["date"]]).drop_duplicates()
    date_prev_map = _build_date_prev_map(all_dates)

    all_prem = []
    for p in periods:
        prem = compute_daily_premium(
            returns, mkt_cap, top_bottom, p, date_prev_map, WEIGHT_METHOD
        )
        for d, v in prem.items():
            all_prem.append((d, v))

    if not all_prem:
        return pd.Series(dtype=float)
    return pd.Series({d: v for d, v in all_prem}).sort_index()


def run_all_factors_backtest(
    factors: pd.DataFrame,
    returns: pd.DataFrame,
    mkt_cap: pd.DataFrame,
    factor_names: list = None,
) -> pd.DataFrame:
    """
    对所有因子运行回测。

    输入:
        factors: 列 (date, code, factor1, factor2, ...)
        returns: 列 (date, code, ret)
        mkt_cap: 列 (date, code, mkt_cap)
        factor_names: 可选，指定因子列名列表
    输出: DataFrame，index=日期，columns=各因子名，value=日频 Top-Bottom 收益差
    """
    names_requested = factor_names or FACTOR_COLUMNS
    names = [c for c in names_requested if c in factors.columns]
    names_missing = [c for c in names_requested if c not in factors.columns]
    if names_missing:
        print(f"  [提示] 以下因子在 factors 表中不存在，已跳过回测: {names_missing}")
    results = {}
    for fn in names:
        s = run_single_factor_backtest(factors, returns, mkt_cap, fn)
        if not s.empty:
            results[fn] = s
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


def _load_phases(path: Path | None, config=default_config) -> list[dict[str, Any]]:
    source = path if path is not None else config.DEFAULT_PHASES_JSON
    if source is not None and Path(source).is_file():
        raw = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        raw = [
            {"label": "2024年9月", "start": "2024-09-06", "end": "2024-09-30"},
            {"label": "2025年6—7月", "start": "2025-06-01", "end": "2025-07-31"},
            {"label": "2025年8—9月", "start": "2025-08-01", "end": "2025-09-30"},
        ]
    result = []
    for i, phase in enumerate(raw):
        if not isinstance(phase, dict):
            raise ValueError(f"阶段 #{i} 须为对象，含 label/start/end")
        result.append({"label": str(phase.get("label", f"阶段{i + 1}")).strip(),
                       "start": str(phase["start"]).strip(), "end": str(phase["end"]).strip()})
    if not result:
        raise ValueError("阶段列表为空")
    return result


def _build_phase_wide(
    premium: pd.DataFrame, factor_cols: list[str], phases: list[dict[str, Any]]
) -> pd.DataFrame:
    frame = premium.copy()
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    rows, labels = [], []
    for phase in phases:
        start, end = pd.Timestamp(phase["start"]), pd.Timestamp(phase["end"])
        if start > end:
            raise ValueError(f'阶段「{phase["label"]}」start 晚于 end')
        part = frame.loc[(frame.index >= start) & (frame.index <= end), factor_cols]
        rows.append({factor: (float((1 + part[factor].dropna().astype(float)).prod() - 1)
                              if len(part[factor].dropna()) else float("nan"))
                     for factor in factor_cols})
        labels.append(phase["label"])
    return pd.DataFrame(rows, index=labels)


def _phase_extremes(
    wide: pd.DataFrame,
    phases: list[dict[str, Any]],
    n: int,
    corr_df: pd.DataFrame | None,
    corr_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, phase in enumerate(phases):
        label, values = str(phase["label"]).strip(), wide.iloc[i].dropna()
        if len(values) < 2:
            print(f"[警告] 阶段「{label}」有效因子不足 2 个，极值表跳过该阶段")
            continue
        try:
            top, bottom, skipped_top, skipped_bottom = pick_top_bottom_from_series(
                values, n, corr_df, corr_threshold
            )
        except ValueError as exc:
            print(f"[警告] 阶段「{label}」极值选取失败: {exc}")
            continue
        for group, selected, skipped in (("Top", top, skipped_top), ("Bottom", bottom, skipped_bottom)):
            for rank, factor in enumerate(selected, 1):
                rows.append({"阶段": label, "极值组": group, "名次": rank, "因子": factor,
                             "区间累计多空": float(values[factor]), "状态": "入选"})
            for factor in skipped:
                rows.append({"阶段": label, "极值组": group, "名次": None, "因子": factor,
                             "区间累计多空": float(values[factor]), "状态": "顺延跳过"})
    if not rows:
        return pd.DataFrame(columns=["阶段", "极值组", "名次", "因子", "区间累计多空", "状态"])
    out = pd.DataFrame(rows)
    out["极值组"] = pd.Categorical(out["极值组"], ["Top", "Bottom"], ordered=True)
    out["状态"] = pd.Categorical(out["状态"], ["入选", "顺延跳过"], ordered=True)
    return out.sort_values(["阶段", "极值组", "状态", "名次"], na_position="last").reset_index(drop=True)


class GroupAnalysis:
    """管理一次分组回测、阶段分析及其全部图表。"""

    def __init__(self, io: DataIO, pipeline: FactorPipeline, config=default_config) -> None:
        self.io, self.pipeline, self.config = io, pipeline, config
        self._premium = pd.DataFrame()
        self._stats = pd.DataFrame()

    @property
    def premium(self) -> pd.DataFrame:
        return self._premium

    @property
    def stats(self) -> pd.DataFrame:
        return self._stats

    def _performance_stats(self, premium: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for column in premium.columns:
            values = backtest_stats(premium[column])
            # 新增字段按已保存 premium 的六位口径计算，便于独立复算；旧字段保持未舍入内存口径。
            expanded = backtest_stats(premium[column].round(6))
            fmt = lambda value: f"{value:.6f}" if not np.isnan(value) else "N/A"
            rows.append({"因子": column, "累计收益": fmt(values["cumret"]),
                         "年化收益": fmt(values["ann_ret"]), "夏普比率": fmt(values["sharpe"]),
                         "最大回撤": fmt(expanded["max_drawdown"]), "Calmar比率": fmt(expanded["calmar"]),
                         "胜率": fmt(expanded["win_rate"]), "交易日数": values["n_days"]})
        return pd.DataFrame(rows)

    def run(
        self,
        factors: pd.DataFrame | None = None,
        returns: pd.DataFrame | None = None,
        mkt_cap: pd.DataFrame | None = None,
    ) -> "GroupAnalysis":
        factors = self.io.load_factors() if factors is None else factors
        returns = self.io.load_returns() if returns is None else returns
        mkt_cap = self.io.load_market_cap() if mkt_cap is None else mkt_cap
        industry = self.io.load_industry() if "industry_neutral" in self.config.PREPROCESS_PIPELINE else None
        prepared = self.pipeline.preprocess(
            factors, steps=self.config.PREPROCESS_PIPELINE,
            factor_columns=self.config.FACTOR_COLUMNS, mkt_cap=mkt_cap,
            industry=industry, winsor_mad_scale=self.config.WINSOR_MAD_SCALE,
            size_neutral_exclude=self.config.SIZE_NEUTRAL_EXCLUDE,
        )
        self._premium = run_all_factors_backtest(prepared, returns, mkt_cap, self.config.FACTOR_COLUMNS)
        self._stats = self._performance_stats(self._premium) if not self._premium.empty else pd.DataFrame()
        if not self._premium.empty:
            self._save_results()
        return self

    def _save_results(self) -> None:
        self.config.DATA_BACKTEST.mkdir(parents=True, exist_ok=True)
        self._premium.round(6).to_csv(
            self.config.DATA_BACKTEST / "backtest_premium_daily.csv", encoding="utf-8-sig"
        )
        self._stats.to_csv(
            self.config.DATA_BACKTEST / "backtest_stats.csv", index=False, encoding="utf-8-sig"
        )
        self.export_temporal()

    def export_temporal(self, freqs=None) -> None:
        frequencies = normalize_export_freqs(
            self.config.TEMPORAL_EXPORT_FREQS if freqs is None else freqs
        )
        path = self.config.DATA_BACKTEST / "backtest_premium_daily.csv"
        if not path.exists():
            return
        premium = pd.read_csv(path, index_col=0, parse_dates=True, encoding="utf-8-sig")
        factor_cols = [column for column in self.config.FACTOR_COLUMNS if column in premium.columns]
        for freq in frequencies:
            period_factor = cum_ls_by_period(premium, factor_cols, freq)
            factor_by_period = period_factor.T
            factor_by_period.index.name = "因子"
            factor_by_period.columns.name = None
            factor_by_period.round(6).to_csv(
                self.config.DATA_BACKTEST / f"backtest_factor_by_period_{freq}.csv",
                encoding="utf-8-sig", index_label="因子"
            )
            wide_to_long_sorted(period_factor, "区间累计多空", "时间区间").round(6).to_csv(
                self.config.DATA_BACKTEST / f"backtest_period_by_factor_{freq}.csv",
                encoding="utf-8-sig", index=False
            )

    def run_phases(
        self, phases_config: Path | None = None, premium: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        phases = _load_phases(phases_config, self.config)
        source = premium if premium is not None else self._premium
        if source.empty:
            source = pd.read_csv(
                self.config.DATA_BACKTEST / "backtest_premium_daily.csv",
                index_col=0, parse_dates=True, encoding="utf-8-sig"
            ).sort_index()
        factors = [column for column in self.config.FACTOR_COLUMNS if column in source.columns]
        wide = _build_phase_wide(source, factors, phases)
        long = wide_to_long_sorted(wide, "区间累计多空", "阶段")
        out_dir = self.config.DATA_BACKTEST
        long.round(6).to_csv(out_dir / "backtest_period_by_factor_phases.csv", encoding="utf-8-sig", index=False)
        transposed = wide.T.copy()
        transposed.index.name, transposed.columns.name = "因子", None
        transposed.round(6).to_csv(out_dir / "backtest_factor_by_period_phases.csv", encoding="utf-8-sig", index_label="因子")
        corr = load_correlation_matrix(self.config.FACTOR_CORRELATION_CSV) if self.config.FACTOR_CORRELATION_CSV.exists() else None
        extremes = _phase_extremes(wide, phases, 3, corr, 0.7)
        extremes.round(6).to_csv(out_dir / "backtest_period_by_factor_phases_extremes.csv", encoding="utf-8-sig", index=False)
        self.plot_phases(phases, source, corr)
        return wide, long

    def plot_curves(self) -> None:
        premium = self._premium if not self._premium.empty else pd.read_csv(
            self.config.DATA_BACKTEST / "backtest_premium_daily.csv", index_col=0,
            parse_dates=True, encoding="utf-8-sig"
        )
        plot_curves(premium, fig_path=self.config.DATA_BACKTEST / "backtest_curves.png")

    def plot_extremes_vs_benchmark(self, n: int = 3) -> None:
        premium = self._premium if not self._premium.empty else pd.read_csv(
            self.config.DATA_BACKTEST / "backtest_premium_daily.csv", index_col=0,
            parse_dates=True, encoding="utf-8-sig"
        )
        stats = self._stats if not self._stats.empty else pd.read_csv(
            self.config.DATA_BACKTEST / "backtest_stats.csv", encoding="utf-8-sig"
        )
        corr = load_correlation_matrix(self.config.FACTOR_CORRELATION_CSV) if self.config.FACTOR_CORRELATION_CSV.exists() else None
        top, bottom, _, _ = pick_top_bottom_factors(stats, n, corr, 0.7)
        benchmark, label = load_benchmark_series(premium.index, None)
        plot_extremes_vs_benchmark(premium, top, bottom, benchmark, label,
                                   self.config.DATA_BACKTEST / "plot_extremes_vs_benchmark.png",
                                   corr_threshold=0.7 if corr is not None else None)

    def plot_period_bars(self) -> None:
        premium = self._premium if not self._premium.empty else pd.read_csv(
            self.config.DATA_BACKTEST / "backtest_premium_daily.csv", index_col=0,
            parse_dates=True, encoding="utf-8-sig"
        )
        generate_period_report(premium, self.config.DATA_BACKTEST, self.config.STACKED_BARS_FREQ)

    def plot_phases(
        self,
        phases: list[dict] | None = None,
        premium: pd.DataFrame | None = None,
        corr_df: pd.DataFrame | None = None,
    ) -> None:
        phases = phases or _load_phases(None, self.config)
        source = premium if premium is not None else self._premium
        if source.empty:
            source = pd.read_csv(self.config.DATA_BACKTEST / "backtest_premium_daily.csv",
                                 index_col=0, parse_dates=True, encoding="utf-8-sig")
        benchmark, label = load_benchmark_series(source.index, None)
        # 原绘图函数按 CSV 读取，确保独立 phase 与统一入口语义一致。
        generate_phase_report(self.config.DATA_BACKTEST / "backtest_premium_daily.csv", phases,
                              self.config.DATA_BACKTEST, 3, benchmark, label, corr_df, 0.7)

    def plot_rrg(self) -> None:
        premium = self._premium if not self._premium.empty else pd.read_csv(
            self.config.DATA_BACKTEST / "backtest_premium_daily.csv", index_col=0,
            parse_dates=True, encoding="utf-8-sig"
        )
        try:
            generate_rrg_report(premium, self.config.DATA_BACKTEST / "rrg.png")
        except Exception as exc:
            print(f"\n[提示] RRG 图跳过: {exc}")


def _build_default() -> GroupAnalysis:
    io = DataIO(default_config.WIDE_CSV_DIR, default_config)
    return GroupAnalysis(io, FactorPipeline(io, default_config), default_config)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="因子分组分析与已有 CSV 重绘")
    parser.add_argument("--plot-only", action="store_true", help="只读取已有 CSV 重绘 Group/Period/RRG")
    args = parser.parse_args(argv)
    result = _build_default()
    if args.plot_only:
        result.plot_curves()
        result.plot_extremes_vs_benchmark()
        result.plot_period_bars()
        result.plot_rrg()
        return 0
    result.run()
    if result.premium.empty:
        print("无有效回测结果")
        return 1
    result.plot_curves()
    result.plot_extremes_vs_benchmark()
    result.plot_period_bars()
    result.plot_rrg()
    print(f"因子分组分析完成: {len(result.premium.columns)} 个因子, {len(result.premium)} 个交易日")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
