"""因子计算、合并、预处理与共享数学实现。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .. import project_config as default_config
from .io import DataIO, read_csv_with_encoding_fallback, round_factor_output


class FactorPipeline:
    """组织因子计算、稳定合并及分析前截面预处理。"""

    VALID_PREPROCESS_STEPS = frozenset(
        {"winsor", "zscore", "industry_neutral", "size_neutral"}
    )

    def __init__(self, io: DataIO, config=default_config) -> None:
        self.io = io
        self.config = config

    @staticmethod
    def _exponential_weights(window: int, halflife: int) -> np.ndarray:
        if window <= 0 or halflife <= 0:
            raise ValueError("window/halflife 须为正")
        age = (window - 1) - np.arange(window, dtype=float)
        raw = np.exp(-(np.log(2.0) / halflife) * age)
        return raw / raw.sum()

    @staticmethod
    def _weighted_ols_intercept_slope(
        y: np.ndarray, x: np.ndarray, w: np.ndarray
    ) -> tuple[float, float]:
        if len(y) != len(x) or len(y) != len(w):
            raise ValueError("y、x、w 须等长，且 w[k] 对应第 k 个时点（时间升序：最旧→最新，与 exponential_weights 对齐）")
        sw = w.sum()
        if sw <= 0:
            return float("nan"), float("nan")
        mx = np.sum(w * x) / sw
        my = np.sum(w * y) / sw
        b = np.sum(w * (x - mx) * (y - my)) / (np.sum(w * (x - mx) ** 2) + 1e-18)
        return float(my - b * mx), float(b)

    @staticmethod
    def _cross_section_residual(
        y: np.ndarray,
        x: np.ndarray,
        *,
        min_count: int = 3,
        preserve_invalid: bool = False,
        catch_errors: bool = True,
    ) -> np.ndarray:
        """唯一 OLS 残差核心；参数保留因子与预处理的旧无效位置语义。"""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < min_count:
            return np.full_like(y, np.nan, dtype=float)
        design = np.column_stack([np.ones(valid.sum()), x[valid]])
        try:
            coef, _, _, _ = np.linalg.lstsq(design, y[valid], rcond=None)
        except Exception:
            if catch_errors:
                return np.full_like(y, np.nan, dtype=float)
            raise
        if preserve_invalid:
            out = np.full_like(y, np.nan, dtype=float)
            out[np.where(valid)[0]] = y[valid] - design @ coef
            return out
        return y - (coef[0] + coef[1] * x)

    @staticmethod
    def _ols_slope(y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(y)
        if valid.sum() < 3:
            return float("nan")
        values = y[valid]
        time = np.arange(len(values), dtype=float)
        time -= time.mean()
        centered = values - values.mean()
        denominator = float(np.dot(time, time))
        if denominator < 1e-18:
            return float("nan")
        return float(np.dot(time, centered) / denominator)

    @staticmethod
    def _rolling_std_mean_ratio(x: np.ndarray) -> float:
        values = np.asarray(x, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            return float("nan")
        mean = float(np.mean(values))
        if abs(mean) < 1e-18:
            return float("nan")
        return float(np.std(values, ddof=0)) / mean

    @staticmethod
    def _five_year_regression_scaled(
        values: pd.Series,
        periods: int = 20,
        *,
        min_points: int | None = None,
    ) -> float:
        valid = values.dropna().astype(float)
        required = max(8, periods // 2) if min_points is None else min_points
        if len(valid) < required:
            return float("nan")
        tail = valid.iloc[-periods:].values if len(valid) >= periods else valid.values
        if len(tail) < 4:
            return float("nan")
        slope = FactorPipeline._ols_slope(tail)
        mean = float(np.nanmean(tail))
        if abs(mean) < 1e-18:
            return float("nan")
        return slope / abs(mean)

    @staticmethod
    def _winsor_series_mad(
        s: pd.Series,
        scale: float = 3.0,
        *,
        n: float | None = None,
        min_count: int = 0,
        coerce_numeric: bool = False,
    ) -> pd.Series:
        """唯一 MAD 核心；显式参数兼容因子计算和分析预处理两种旧口径。"""
        if n is not None:
            scale = n
        series = pd.to_numeric(s, errors="coerce") if coerce_numeric else s
        valid = series.dropna()
        if len(valid) < min_count:
            return series
        median = valid.median()
        mad = (valid - median).abs().median()
        if pd.isna(mad) or mad <= 0:
            return series
        low, high = median - scale * mad, median + scale * mad
        if low >= high:
            return series
        return series.clip(low, high)

    @staticmethod
    def _cross_section_zscore(
        df: pd.DataFrame, value_cols: list[str], date_col: str = "date"
    ) -> pd.DataFrame:
        out = df.copy()
        for column in value_cols:
            if column not in out.columns:
                continue

            def _z(series: pd.Series) -> pd.Series:
                series = pd.to_numeric(series, errors="coerce")
                if series.notna().sum() < 2:
                    return pd.Series(np.nan, index=series.index)
                mean = float(series.mean())
                std = series.std(ddof=0)
                if std is None or not np.isfinite(std) or float(std) == 0.0:
                    return (series * 0.0).where(series.notna(), np.nan)
                return (series - mean) / float(std)

            out[f"{column}__z"] = out.groupby(date_col)[column].transform(_z)
        return out

    @staticmethod
    def _get_level1_merge_config(
        style_key: str, descriptors: Sequence[str], config=default_config
    ):
        weights = config.LEVEL1_WEIGHTED.get(style_key)
        if weights is None or len(weights) != len(descriptors):
            raise ValueError(f"LEVEL1_WEIGHTED[{style_key!r}] 与 descriptors 不一致")
        return config.Level1MergeConfig(
            weights={name: float(weights[i]) for i, name in enumerate(descriptors)},
            method=config.LEVEL1_METHOD.get(style_key, "zscore_weighted"),
        )

    @staticmethod
    def _merge_descriptors_to_level1(
        df: pd.DataFrame,
        descriptor_cols: list[str],
        style_key: str,
        style_col: str,
        *,
        date_col: str = "date",
        method=None,
    ) -> pd.DataFrame:
        cfg = FactorPipeline._get_level1_merge_config(style_key, descriptor_cols)
        merge_method = cfg.method if method is None else method
        use = [column for column in descriptor_cols if column in df.columns]
        if not use:
            out = df.copy()
            out[style_col] = np.nan
            return out
        weights = {column: cfg.weights[column] for column in use}
        if merge_method == "weighted_raw":
            out = df.copy()
            acc = np.zeros(len(out), dtype=float)
            for column in use:
                acc += weights[column] * pd.to_numeric(out[column], errors="coerce").fillna(0.0).values
            out[style_col] = acc
            return out
        out = FactorPipeline._cross_section_zscore(df, use, date_col=date_col)
        acc = np.zeros(len(out), dtype=float)
        weight_sum = np.zeros(len(out), dtype=float)
        for column in use:
            zcolumn = f"{column}__z"
            if zcolumn not in out.columns:
                continue
            values = pd.to_numeric(out[zcolumn], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values)
            weight = float(weights[column])
            acc += np.where(finite, weight * values, 0.0)
            weight_sum += np.where(finite, weight, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[style_col] = np.where(weight_sum > 0.0, acc / weight_sum, np.nan)
        return out.drop(columns=[c for c in out.columns if c.endswith("__z")], errors="ignore")

    def compute_one(self, factor: str | type, run_target=None) -> pd.DataFrame:
        """计算并保存一个因子，不触发合并。"""
        from .factors import FACTOR_CLASS_BY_NAME

        factor_class = FACTOR_CLASS_BY_NAME[factor] if isinstance(factor, str) else factor
        instance = factor_class(self.io)
        result = instance.compute(run_target)
        path = Path(self.config.DATA_FACTORS) / instance.output_file
        path.parent.mkdir(parents=True, exist_ok=True)
        round_factor_output(result, self.config.FACTOR_OUTPUT_DECIMALS).to_csv(
            path, index=False, encoding="utf-8-sig"
        )
        print(f"输出: {path}，记录数 {len(result)}")
        return result

    def compute_all(self) -> dict[str, pd.DataFrame]:
        """按固定顺序运行默认 16 个 Barra 因子；失败立即向上抛出。"""
        from .factors import BARRA_FACTOR_CLASSES

        results: dict[str, pd.DataFrame] = {}
        for factor_class in BARRA_FACTOR_CLASSES:
            results[factor_class.name] = self.compute_one(factor_class)
        return results

    def _select_factor_columns(self, df: pd.DataFrame, factor_class: type) -> pd.DataFrame:
        if not all(column in df.columns for column in ("date", "code")):
            return df
        style_key, descriptors = factor_class.name, factor_class.descriptors
        if style_key in df.columns:
            return df[["date", "code", style_key]].copy()
        present = [column for column in df.columns if column in set(descriptors)]
        return df[["date", "code"] + present].copy() if present else df

    def merge(self, results: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
        """按旧文件顺序合并已保存因子，保存合并表与相关矩阵。"""
        from .factors import ALL_FACTOR_CLASSES

        merged: pd.DataFrame | None = None
        Path(self.config.DATA_FACTORS).mkdir(parents=True, exist_ok=True)
        for factor_class in ALL_FACTOR_CLASSES:
            path = Path(self.config.DATA_FACTORS) / factor_class.output_file
            if not path.exists():
                continue
            frame = read_csv_with_encoding_fallback(path)
            frame["date"] = pd.to_datetime(frame["date"])
            frame = frame[(frame["date"] >= self.config.ANALYSIS_START) & (frame["date"] <= self.config.ANALYSIS_END)]
            frame = self._select_factor_columns(frame, factor_class)
            merged = frame if merged is None else merged.merge(frame, on=["date", "code"], how="outer")
        if merged is None:
            print("\n[合并] 无可用因子文件，跳过。")
            return pd.DataFrame()
        merged = merged.sort_values(["date", "code"]).reset_index(drop=True)
        keys = ["date", "code"]
        merged = merged[keys + [c for c in merged.columns if c not in keys]]
        Path(self.config.DATA_OUT).mkdir(parents=True, exist_ok=True)
        round_factor_output(merged, self.config.FACTOR_OUTPUT_DECIMALS).to_csv(
            self.config.FACTORS_MERGED, index=False, encoding="utf-8-sig"
        )
        print(f"\n[合并] 已输出 {self.config.FACTORS_MERGED}，共 {len(merged)} 行，日期范围 {self.config.ANALYSIS_START} ~ {self.config.ANALYSIS_END}")
        numeric = [c for c in merged.columns if c not in keys and pd.api.types.is_numeric_dtype(merged[c])]
        if numeric:
            corr = merged[numeric].corr()
            corr.round(self.config.FACTOR_OUTPUT_DECIMALS).to_csv(
                self.config.FACTOR_CORRELATION_CSV, encoding="utf-8-sig"
            )
            print(f"[相关性] 已输出 {self.config.FACTOR_CORRELATION_CSV}（{len(corr)}×{len(corr)}，分析期内 date×code 长表 Pearson）")
        return merged

    def preprocess(
        self,
        factors: pd.DataFrame,
        *,
        steps: Sequence[str] | None = None,
        factor_columns: Sequence[str] | None = None,
        mkt_cap: Optional[pd.DataFrame] = None,
        industry: Optional[pd.DataFrame] = None,
        winsor_mad_scale: float | None = None,
        size_neutral_exclude: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """按显式步骤顺序执行一次截面预处理。"""
        step_list = [step for step in (steps if steps is not None else self.config.PREPROCESS_PIPELINE) if step]
        unknown = set(step_list) - self.VALID_PREPROCESS_STEPS
        if unknown:
            raise ValueError(f"未知预处理步骤 {unknown}，允许: {sorted(self.VALID_PREPROCESS_STEPS)}")
        requested = factor_columns if factor_columns is not None else self.config.FACTOR_COLUMNS
        columns = [column for column in requested if column in factors.columns]
        if not columns:
            return factors.copy()
        work = factors.copy()
        scale = self.config.WINSOR_MAD_SCALE if winsor_mad_scale is None else winsor_mad_scale
        exclude = tuple(self.config.SIZE_NEUTRAL_EXCLUDE if size_neutral_exclude is None else size_neutral_exclude)
        for step in step_list:
            if step == "winsor":
                for column in columns:
                    work[column] = work.groupby("date", group_keys=False)[column].transform(
                        lambda series, sc=scale: self._winsor_series_mad(
                            series, sc, min_count=5, coerce_numeric=True
                        )
                    )
            elif step == "zscore":
                for column in columns:
                    work[column] = work.groupby("date", group_keys=False)[column].transform(
                        self._preprocess_zscore
                    )
            elif step == "industry_neutral":
                work = self._industry_neutral(work, columns, industry)
            elif step == "size_neutral":
                work = self._size_neutral(work, columns, mkt_cap, exclude)
        return work

    @staticmethod
    def _preprocess_zscore(series: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        mean, std = values.mean(), values.std(ddof=0)
        if std == 0 or not np.isfinite(std) or values.notna().sum() < 2:
            return pd.Series(np.nan, index=values.index)
        return (values - mean) / std

    @staticmethod
    def _industry_neutral(
        df: pd.DataFrame, columns: list[str], industry: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        if industry is None or industry.empty:
            raise ValueError("已启用 industry_neutral，但未加载行业数据。")
        work = df.merge(industry[["date", "code", "industry"]], on=["date", "code"], how="left")
        work["industry"] = work["industry"].fillna("_缺失")
        for column in columns:
            mean = work.groupby(["date", "industry"])[column].transform("mean")
            work[column] = pd.to_numeric(work[column], errors="coerce") - mean
        return work.drop(columns=["industry"])

    @staticmethod
    def _size_neutral(
        df: pd.DataFrame,
        columns: list[str],
        mkt_cap: Optional[pd.DataFrame],
        exclude: Sequence[str],
    ) -> pd.DataFrame:
        if mkt_cap is None or mkt_cap.empty:
            raise ValueError("已启用 size_neutral 但未提供 mkt_cap。")
        market_cap = mkt_cap[["date", "code", "mkt_cap"]].copy()
        market_cap["date"] = pd.to_datetime(market_cap["date"])
        market_cap["_log_mcap"] = np.log(np.maximum(market_cap["mkt_cap"].astype(float), 1.0))
        work = df.merge(market_cap[["date", "code", "_log_mcap"]], on=["date", "code"], how="left")
        skip = set(exclude)
        for column in columns:
            if column in skip or column not in work.columns:
                continue
            pieces = []
            for _, group in work.groupby("date", sort=False):
                residual = FactorPipeline._cross_section_residual(
                    pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float),
                    group["_log_mcap"].to_numpy(dtype=float),
                    min_count=3,
                    preserve_invalid=True,
                    catch_errors=False,
                )
                pieces.append(pd.Series(residual, index=group.index))
            work[column] = pd.concat(pieces)
        return work.drop(columns=["_log_mcap"])

    def run(self, *, preprocess: bool = False) -> pd.DataFrame:
        """完整因子链路；默认保留 factors-only 的未预处理合并表语义。"""
        results = self.compute_all()
        merged = self.merge(results)
        return self.preprocess(merged) if preprocess else merged
