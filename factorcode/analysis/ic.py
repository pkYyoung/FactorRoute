"""IC 计算、切片、保存与绘图。"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from .. import project_config as default_config
from ..data.io import DataIO
from ..data.pipeline import FactorPipeline
from .common import compute_period_avg, distinct_line_colors, mean_ic_by_period, normalize_export_freqs, wide_to_long_sorted
from .metrics import ic_single_period, ic_stats

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

_LINE_MARKERS = ("o", "s", "^", "D", "v", "p", "P", "X", "*", "h", "8", "H", "d", "<", ">", "1", "2", "3", "4")


class ICAnalysis:
    """管理一次 IC 分析的内存结果及对应输出。"""

    def __init__(self, io: DataIO, pipeline: FactorPipeline, config=default_config) -> None:
        self.io, self.pipeline, self.config = io, pipeline, config
        self._series = pd.DataFrame()
        self._stats = pd.DataFrame()

    @property
    def series(self) -> pd.DataFrame:
        return self._series

    @property
    def stats(self) -> pd.DataFrame:
        return self._stats

    def _period_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        frame = returns.copy()
        frame["period"] = pd.to_datetime(frame["date"]).dt.to_period(self.config.PERIOD_FREQ)
        result = frame.groupby(["period", "code"])["ret"].apply(lambda x: (1 + x).prod() - 1).reset_index()
        return result.rename(columns={"ret": "ret_period"})

    def _single_factor_ic(
        self, factor_avg: pd.DataFrame, period_returns: pd.DataFrame, factor_col: str
    ) -> pd.Series:
        values = []
        return_periods = set(period_returns["period"].unique())
        for period in sorted(factor_avg["period"].unique()):
            next_period = period + 1
            if next_period not in return_periods:
                continue
            current = factor_avg[factor_avg["period"] == period][["code", factor_col]]
            following = period_returns[period_returns["period"] == next_period][["code", "ret_period"]]
            merged = current.merge(following, on="code", how="inner").dropna()
            if len(merged) < 3:
                continue
            value = ic_single_period(merged[factor_col], merged["ret_period"], self.config.IC_CORR_TYPE)
            values.append((period, value))
        return pd.Series({period: value for period, value in values}).sort_index() if values else pd.Series(dtype=float)

    def _compute_series(self, factors: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        requested = self.config.FACTOR_COLUMNS
        names = [name for name in requested if name in factors.columns]
        missing = [name for name in requested if name not in factors.columns]
        if missing:
            print(f"  [提示] 以下因子在 factors 中不存在，无法进行 IC 计算: {missing}")
        period_returns = self._period_returns(returns)
        result = {}
        for name in names:
            average = compute_period_avg(factors, name, self.config.PERIOD_FREQ)
            series = self._single_factor_ic(average, period_returns, f"{name}_avg")
            if not series.empty:
                result[name] = series
        return pd.DataFrame(result)

    @staticmethod
    def _compute_stats(series: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for column in series.columns:
            values = ic_stats(series[column])
            rows.append({"因子": column, "IC均值": values["ic_mean"],
                         "IC标准差": values["ic_std"], "IC_IR": values["ic_ir"],
                         "IC>0比例": values["ic_pos_pct"], "|IC|均值": values["ic_abs_mean"],
                         "周期数": values["n_periods"]})
        return pd.DataFrame(rows)

    def _save_results(self) -> None:
        self.config.DATA_IC.mkdir(parents=True, exist_ok=True)
        out_series = self.config.DATA_IC / "ic_series.csv"
        save_series = self._series.copy()
        save_series.index = save_series.index.astype(str)
        save_series.round(6).to_csv(out_series, encoding="utf-8-sig")
        self._stats.to_csv(self.config.DATA_IC / "ic_stats.csv", index=False, encoding="utf-8-sig")
        self.export_temporal()

    def run(
        self, factors: pd.DataFrame | None = None, returns: pd.DataFrame | None = None
    ) -> "ICAnalysis":
        factors = self.io.load_factors() if factors is None else factors
        returns = self.io.load_returns() if returns is None else returns
        industry = self.io.load_industry() if "industry_neutral" in self.config.PREPROCESS_PIPELINE else None
        market_cap = self.io.load_market_cap() if "size_neutral" in self.config.PREPROCESS_PIPELINE else None
        prepared = self.pipeline.preprocess(
            factors,
            steps=self.config.PREPROCESS_PIPELINE,
            factor_columns=self.config.FACTOR_COLUMNS,
            mkt_cap=market_cap,
            industry=industry,
            winsor_mad_scale=self.config.WINSOR_MAD_SCALE,
            size_neutral_exclude=self.config.SIZE_NEUTRAL_EXCLUDE,
        )
        self._series = self._compute_series(prepared, returns)
        self._stats = self._compute_stats(self._series) if not self._series.empty else pd.DataFrame()
        if not self._series.empty:
            self._save_results()
        return self

    def export_temporal(self, freqs=None) -> None:
        frequencies = normalize_export_freqs(
            self.config.TEMPORAL_EXPORT_FREQS if freqs is None else freqs
        )
        path = self.config.DATA_IC / "ic_series.csv"
        if not path.exists():
            return
        source = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
        factor_cols = [c for c in self.config.FACTOR_COLUMNS if c in source.columns]
        for freq in frequencies:
            period_factor = mean_ic_by_period(
                source, factor_cols, freq, self.config.TEMPORAL_IC_LABEL_ANCHOR
            )
            factor_by_period = period_factor.T
            factor_by_period.index.name = "因子"
            factor_by_period.columns.name = None
            factor_by_period.round(6).to_csv(
                self.config.DATA_IC / f"ic_factor_by_period_{freq}.csv",
                encoding="utf-8-sig", index_label="因子"
            )
            long = wide_to_long_sorted(period_factor, "区间均值IC", "时间区间")
            long.round(6).to_csv(
                self.config.DATA_IC / f"ic_period_by_factor_{freq}.csv",
                encoding="utf-8-sig", index=False
            )

    def plot_curves(
        self, source: pd.DataFrame | None = None, fig_path: Path | None = None,
        figsize: tuple = (12, 6), max_factors: int = 8
    ) -> None:
        df = source if source is not None else self._series
        if df.empty:
            path = self.config.DATA_IC / "ic_series.csv"
            if not path.exists():
                print(f"未找到 IC 数据: {path}，请先运行 python -m factorcode.research_engine ic")
                return
            df = pd.read_csv(path, index_col=0, encoding="utf-8-sig")
        cols = [c for c in df.columns if df[c].notna().any()]
        if not cols:
            print("无有效 IC 数据可绘制")
            return
        fig_path = fig_path or self.config.DATA_IC / "ic_curves.png"
        x_dates = pd.Index(df.index).map(
            lambda value: pd.to_datetime(str(value).split("/")[1]) if "/" in str(value) else pd.to_datetime(value)
        )
        plot_df = df[cols].cumsum()
        colors = distinct_line_colors(len(cols))
        markers = [_LINE_MARKERS[i % len(_LINE_MARKERS)] for i in range(len(cols))]
        color_by_col, marker_by_col = dict(zip(cols, colors)), dict(zip(cols, markers))
        if len(cols) <= max_factors:
            fig, ax = plt.subplots(figsize=figsize)
            axes = [ax]
            groups = [cols]
        else:
            n_rows = (len(cols) + max_factors - 1) // max_factors
            fig, raw_axes = plt.subplots(n_rows, 1, figsize=(figsize[0], figsize[1] * n_rows), sharex=True)
            axes = list(raw_axes) if n_rows > 1 else [raw_axes]
            groups = [cols[i * max_factors:(i + 1) * max_factors] for i in range(n_rows)]
        for ax, group in zip(axes, groups):
            for col in group:
                ax.plot(x_dates, plot_df[col].values, label=col, color=color_by_col[col],
                        marker=marker_by_col[col], markersize=4, linewidth=1.8, alpha=0.9)
            ax.set_ylabel("累计 IC")
            ax.legend(loc="best", fontsize=9)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("周期")
        if len(cols) <= max_factors:
            axes[0].set_title("IC 回测：各因子累计 IC 序列")
        else:
            fig.suptitle("IC 回测：各因子累计 IC 序列", fontsize=12, y=1.02)
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"IC 曲线图已保存: {fig_path}")
        plt.close()


def _build_default() -> ICAnalysis:
    io = DataIO(default_config.WIDE_CSV_DIR, default_config)
    return ICAnalysis(io, FactorPipeline(io, default_config), default_config)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="IC 分析与已有 CSV 重绘")
    parser.add_argument("--plot-only", action="store_true", help="只读取已有 ic_series.csv 重绘")
    args = parser.parse_args(argv)
    result = _build_default()
    if args.plot_only:
        result.plot_curves()
        return 0
    result.run()
    if result.series.empty:
        print("无有效 IC 结果")
        return 1
    result.plot_curves()
    print(f"IC 分析完成: {len(result.series.columns)} 个因子, {len(result.series)} 个周期")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
