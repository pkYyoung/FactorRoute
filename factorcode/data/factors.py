"""16 个 Barra 风格因子的统一类族。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
import pandas as pd

from ..project_config import (
    ATVR_HALFLIFE, ATVR_WINDOW, DATA_FACTORS, DASTD_HALFLIFE, DASTD_WINDOW,
    HBETA_HALFLIFE, HBETA_WINDOW, LTRSTR_HALFLIFE, LTRSTR_LAG, LTRSTR_WINDOW,
    RSTR_HALFLIFE, RSTR_WINDOW, STOA_DAYS, STOQ_DAYS, STOM_DAYS, RunTarget,
    get_target_for_style,
)
from .io import DataIO, read_csv_with_encoding_fallback
from .pipeline import FactorPipeline


def _resolve_factor_target(
    name: str, descriptors: tuple[str, ...], run_target: RunTarget | None
) -> tuple[str, list[str]]:
    value = get_target_for_style(name) if run_target is None else str(run_target).strip()
    if value == name:
        return "level1", list(descriptors)
    if value in descriptors:
        return "level2", [value]
    raise ValueError(f"{name} 的目标 {value!r} 无效：填一级名 {name!r} 或二级名之一 {list(descriptors)}")


class BarraFactor(ABC):
    """定义 Barra 因子的统一计算协议和运行目标语义。"""

    name: str
    descriptors: tuple[str, ...]
    output_file: str

    def __init__(self, io: DataIO) -> None:
        self.io = io

    def compute(
        self,
        run_target: RunTarget | None = None,
        *,
        merge_style: bool = False,
    ) -> pd.DataFrame:
        """按一级或指定二级运行目标计算因子，并返回未舍入的内存结果。"""
        kind, descriptors = _resolve_factor_target(self.name, self.descriptors, run_target)
        df = self._assemble_descriptors(kind, descriptors)
        if kind == "level1" or merge_style:
            df = self._compute_level1(df)
        return df

    @abstractmethod
    def _assemble_descriptors(self, kind: str, descriptors: list[str]) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def _compute_level1(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class SizeFactor(BarraFactor):
    """计算规模因子 Size 及其 LNCAP 描述子。"""

    name = 'Size'
    descriptors = ('LNCAP',)
    output_file = 'style_size_long.csv'
    _descriptor_cols = ['LNCAP']

    def _compute_lncap(self) -> pd.DataFrame:
        df = self.io.read_metric('总市值')
        df = self.io.apply_missing_rules(df, '总市值')
        df = df.rename(columns={'value': 'mkt_cap'})
        df['LNCAP'] = np.log(pd.to_numeric(df['mkt_cap'], errors='coerce').clip(lower=1.0))
        return df[['date', 'code', 'LNCAP']].sort_values(['date', 'code']).reset_index(drop=True)

    def _compute_level1(self, df: pd.DataFrame) -> pd.DataFrame:
        return FactorPipeline._merge_descriptors_to_level1(df, self._descriptor_cols, self.name, self.name, date_col='date')

    def _assemble_descriptors(self, kind: str, descriptors: list[str]) -> pd.DataFrame:
        need = set(self._descriptor_cols) if kind == 'level1' else set(descriptors)
        if 'LNCAP' in need:
            return self._compute_lncap()
        return pd.DataFrame(columns=['date', 'code'] + self._descriptor_cols)

class BetaFactor(BarraFactor):
    """计算市场敏感度因子 Beta 及其 HBETA 描述子。"""

    name = 'Beta'
    descriptors = ('HBETA',)
    output_file = 'style_beta_long.csv'
    _descriptor_cols = ['HBETA']
    W_HBETA = FactorPipeline._exponential_weights(HBETA_WINDOW, HBETA_HALFLIFE)

    def _compute_hbeta(self) -> pd.DataFrame:
        df = self.io.read_metric('收盘价')
        df = self.io.apply_missing_rules(df, '收盘价')
        df = df.rename(columns={'value': 'price'})
        df = df.sort_values(['code', 'date'])
        df['ret'] = df.groupby('code')['price'].pct_change()
        df = df.dropna(subset=['ret'])
        index_ret = self.io.load_benchmark_returns()
        results = []
        for (code, g) in df.groupby('code', group_keys=False):
            g = g.sort_values('date').set_index('date')[['ret']]
            g = g.join(index_ret.to_frame('index_ret'), how='inner').dropna()
            if len(g) < HBETA_WINDOW:
                continue
            arr = g.values
            idx = g.index
            for i in range(HBETA_WINDOW - 1, len(arr)):
                y = arr[i - HBETA_WINDOW + 1:i + 1, 0]
                x = arr[i - HBETA_WINDOW + 1:i + 1, 1]
                if np.any(np.isnan(x)) or np.any(np.isnan(y)):
                    continue
                (_, beta) = FactorPipeline._weighted_ols_intercept_slope(y, x, self.W_HBETA)
                if not np.isfinite(beta):
                    continue
                results.append((idx[i], code, beta))
        if not results:
            return pd.DataFrame(columns=['date', 'code', 'HBETA'])
        out = pd.DataFrame(results, columns=['date', 'code', 'HBETA'])
        return out.sort_values(['date', 'code']).reset_index(drop=True)

    def _compute_level1(self, df: pd.DataFrame) -> pd.DataFrame:
        return FactorPipeline._merge_descriptors_to_level1(df, self._descriptor_cols, self.name, self.name, date_col='date')

    def _assemble_descriptors(self, kind: str, descriptors: list[str]) -> pd.DataFrame:
        need = set(self._descriptor_cols) if kind == 'level1' else set(descriptors)
        if 'HBETA' in need:
            return self._compute_hbeta()
        return pd.DataFrame(columns=['date', 'code'] + self._descriptor_cols)

class LiquidityFactor(BarraFactor):
    """计算流动性因子及不同窗口的换手率描述子。"""

    name = 'Liquidity'
    descriptors = ('STOM', 'STOQ', 'STOA', 'ATVR')
    output_file = 'style_liquidity_long.csv'
    _descriptor_cols = ['STOM', 'STOQ', 'STOA', 'ATVR']
    W_ATVR = FactorPipeline._exponential_weights(ATVR_WINDOW, ATVR_HALFLIFE)
    _ROLLING_SPECS: tuple[tuple[int, str], ...] = ((STOM_DAYS, 'STOM'), (STOQ_DAYS, 'STOQ'), (STOA_DAYS, 'STOA'))

    def _load_daily_turnover(self) -> pd.DataFrame:
        vol = self.io.read_metric('成交量')
        vol = self.io.apply_missing_rules(vol, '成交量')
        vol = vol.rename(columns={'value': 'vol'})
        flo = self.io.read_metric('流通股本')
        flo = self.io.apply_missing_rules(flo, '流通股本')
        flo = flo.rename(columns={'value': 'float_shares'})
        df = vol.merge(flo, on=['date', 'code'], how='inner')
        df = df[(df['float_shares'] > 0) & (df['vol'] >= 0)]
        df['turnover_rate'] = df['vol'] / df['float_shares']
        return df[['date', 'code', 'turnover_rate']].sort_values(['code', 'date'])

    def _add_rolling_liquidity_cols(self, d: pd.DataFrame, want: set[str]) -> pd.DataFrame:
        """对单只股票日序列：按 want 中列名，用对应窗口做 sum(日换手) 再 ln。"""
        d = d.sort_values('date')
        tr = pd.to_numeric(d['turnover_rate'], errors='coerce')
        tr = tr.where(tr <= 1.0, tr / 100.0)
        for (days, col) in self._ROLLING_SPECS:
            if col not in want:
                continue
            s = tr.rolling(window=days, min_periods=days).sum()
            d[col] = np.log(s.clip(lower=1e-12))
        return d

    def _atvr(self, d: pd.DataFrame) -> pd.DataFrame:
        d = d.sort_values('date')
        tr = pd.to_numeric(d['turnover_rate'], errors='coerce')
        tr = tr.where(tr <= 1.0, tr / 100.0)

        def _ewm_sum(x: np.ndarray) -> float:
            if len(x) < ATVR_WINDOW:
                return float('nan')
            seg = x[-ATVR_WINDOW:]
            return float(np.dot(self.W_ATVR, seg))
        vals = []
        arr = tr.values
        for i in range(len(d)):
            vals.append(_ewm_sum(arr[:i + 1]))
        d['ATVR'] = vals
        return d

    def _compute_stom(self) -> pd.DataFrame:
        daily = self._load_daily_turnover()
        parts = []
        for (code, g) in daily.groupby('code'):
            parts.append(self._add_rolling_liquidity_cols(g, {'STOM'}))
        out = pd.concat(parts, ignore_index=True)
        return out[['date', 'code', 'STOM']].dropna(subset=['STOM']).sort_values(['date', 'code']).reset_index(drop=True)

    def _compute_stoq(self) -> pd.DataFrame:
        daily = self._load_daily_turnover()
        parts = []
        for (code, g) in daily.groupby('code'):
            parts.append(self._add_rolling_liquidity_cols(g, {'STOQ'}))
        out = pd.concat(parts, ignore_index=True)
        return out[['date', 'code', 'STOQ']].dropna(subset=['STOQ']).sort_values(['date', 'code']).reset_index(drop=True)

    def _compute_stoa(self) -> pd.DataFrame:
        daily = self._load_daily_turnover()
        parts = []
        for (code, g) in daily.groupby('code'):
            parts.append(self._add_rolling_liquidity_cols(g, {'STOA'}))
        out = pd.concat(parts, ignore_index=True)
        return out[['date', 'code', 'STOA']].dropna(subset=['STOA']).sort_values(['date', 'code']).reset_index(drop=True)

    def _compute_atvr(self) -> pd.DataFrame:
        daily = self._load_daily_turnover()
        parts = []
        for (code, g) in daily.groupby('code'):
            parts.append(self._atvr(g))
        out = pd.concat(parts, ignore_index=True)
        return out[['date', 'code', 'ATVR']].dropna(subset=['ATVR']).sort_values(['date', 'code']).reset_index(drop=True)

    def _assemble_liquidity_full(self) -> pd.DataFrame:
        daily = self._load_daily_turnover()
        parts = []
        want_roll = {'STOM', 'STOQ', 'STOA'}
        for (code, g) in daily.groupby('code'):
            g = self._add_rolling_liquidity_cols(g, want_roll)
            g = self._atvr(g)
            parts.append(g)
        daily = pd.concat(parts, ignore_index=True)
        return daily[['date', 'code', 'STOM', 'STOQ', 'STOA', 'ATVR']].sort_values(['date', 'code']).reset_index(drop=True)

    def _assemble_descriptors(self, kind: str, descriptors: list[str]) -> pd.DataFrame:
        need = set(self._descriptor_cols) if kind == 'level1' else set(descriptors)
        if not need:
            return pd.DataFrame(columns=['date', 'code'] + self._descriptor_cols)
        if need == set(self._descriptor_cols):
            return self._assemble_liquidity_full()
        dfs: list[pd.DataFrame] = []
        if 'STOM' in need:
            dfs.append(self._compute_stom())
        if 'STOQ' in need:
            dfs.append(self._compute_stoq())
        if 'STOA' in need:
            dfs.append(self._compute_stoa())
        if 'ATVR' in need:
            dfs.append(self._compute_atvr())
        if not dfs:
            return pd.DataFrame(columns=['date', 'code'] + self._descriptor_cols)
        out = dfs[0]
        for d in dfs[1:]:
            out = out.merge(d, on=['date', 'code'], how='outer')
        cols = ['date', 'code'] + [c for c in self._descriptor_cols if c in need]
        return out[cols].sort_values(['date', 'code']).reset_index(drop=True)

    def _compute_level1(self, df: pd.DataFrame) -> pd.DataFrame:
        return FactorPipeline._merge_descriptors_to_level1(df, self._descriptor_cols, self.name, self.name, date_col='date')

class MomentumFactor(BarraFactor):
    """计算动量因子及 RSTR、HALPHA 描述子。"""

    name = 'Momentum'
    descriptors = ('RSTR', 'HALPHA')
    output_file = 'style_momentum_long.csv'
    _descriptor_cols = ['RSTR', 'HALPHA']
    W_RSTR = FactorPipeline._exponential_weights(RSTR_WINDOW, RSTR_HALFLIFE)
    W_BETA = FactorPipeline._exponential_weights(HBETA_WINDOW, HBETA_HALFLIFE)

    def _price_with_mret(self) -> pd.DataFrame:
        daily = self.io.read_metric('收盘价')
        daily = self.io.apply_missing_rules(daily, '收盘价')
        daily = daily.rename(columns={'value': 'price'})
        daily = daily.sort_values(['code', 'date'])
        daily['price'] = pd.to_numeric(daily['price'], errors='coerce')
        daily['ret'] = daily.groupby('code')['price'].pct_change()
        daily = daily.dropna(subset=['ret'])
        mret = self.io.load_benchmark_returns()
        daily = daily.merge(mret.rename('mret').reset_index(), on='date', how='inner')
        daily['logret'] = np.log1p(daily['ret'].clip(lower=-0.9999))
        return daily

    def _compute_rstr(self) -> pd.DataFrame:
        daily = self._price_with_mret()
        rows: list[tuple] = []
        need = max(RSTR_WINDOW, 22)
        for (code, g) in daily.groupby('code'):
            g = g.sort_values('date')
            if len(g) < need + 10:
                continue
            lr = g['logret'].values.astype(float)
            dates = g['date'].values
            n = len(g)
            rstr_raw = np.full(n, np.nan)
            for i in range(n):
                if i >= RSTR_WINDOW - 1:
                    seg = lr[i - RSTR_WINDOW + 1:i + 1]
                    if np.all(np.isfinite(seg)):
                        rstr_raw[i] = float(np.sum(self.W_RSTR * seg))
            rstr_adj = np.full(n, np.nan)
            for i in range(n):
                if i < 11 + 10:
                    continue
                win = rstr_raw[i - 11 - 10:i - 11 + 1]
                if np.all(np.isfinite(win)):
                    rstr_adj[i] = float(np.nanmean(win))
            for i in range(n):
                if np.isfinite(rstr_adj[i]):
                    rows.append((pd.Timestamp(dates[i]), code, rstr_adj[i]))
        if not rows:
            return pd.DataFrame(columns=['date', 'code', 'RSTR'])
        return pd.DataFrame(rows, columns=['date', 'code', 'RSTR']).sort_values(['date', 'code']).reset_index(drop=True)

    def _compute_halpha(self) -> pd.DataFrame:
        daily = self._price_with_mret()
        rows: list[tuple] = []
        for (code, g) in daily.groupby('code'):
            g = g.sort_values('date')
            if len(g) < HBETA_WINDOW + 5:
                continue
            rets = g['ret'].values.astype(float)
            mx = g['mret'].values.astype(float)
            dates = g['date'].values
            n = len(g)
            for i in range(HBETA_WINDOW - 1, n):
                y = rets[i - HBETA_WINDOW + 1:i + 1]
                x = mx[i - HBETA_WINDOW + 1:i + 1]
                if np.all(np.isfinite(y)) and np.all(np.isfinite(x)):
                    (a, _) = FactorPipeline._weighted_ols_intercept_slope(y, x, self.W_BETA)
                    rows.append((pd.Timestamp(dates[i]), code, a))
        if not rows:
            return pd.DataFrame(columns=['date', 'code', 'HALPHA'])
        return pd.DataFrame(rows, columns=['date', 'code', 'HALPHA']).sort_values(['date', 'code']).reset_index(drop=True)

    def _assemble_descriptors(self, kind: str, descriptors: list[str]) -> pd.DataFrame:
        need = set(self._descriptor_cols) if kind == 'level1' else set(descriptors)
        parts: list[pd.DataFrame] = []
        if 'RSTR' in need:
            parts.append(self._compute_rstr())
        if 'HALPHA' in need:
            parts.append(self._compute_halpha())
        if not parts:
            return pd.DataFrame(columns=['date', 'code'] + self._descriptor_cols)
        out = parts[0]
        for p in parts[1:]:
            out = out.merge(p, on=['date', 'code'], how='outer')
        return out.sort_values(['date', 'code']).reset_index(drop=True)

    def _compute_level1(self, df: pd.DataFrame) -> pd.DataFrame:
        return FactorPipeline._merge_descriptors_to_level1(df, self._descriptor_cols, self.name, self.name, date_col='date')

class ResidualVolatilityFactor(BarraFactor):
    """计算残差波动率因子及 HSIGMA、DASTD、CMRA 描述子。"""

    name = 'ResidualVolatility'
    descriptors = ('HSIGMA', 'DASTD', 'CMRA')
    output_file = 'style_residual_volatility_long.csv'
    _descriptor_cols = ['HSIGMA', 'DASTD', 'CMRA']
    W_DASTD = FactorPipeline._exponential_weights(DASTD_WINDOW, DASTD_HALFLIFE)
    W_HBETA = FactorPipeline._exponential_weights(HBETA_WINDOW, HBETA_HALFLIFE)

    def _weighted_std(self, y: np.ndarray, w: np.ndarray) -> float:
        """y、w 等长且同序（时间升序，与 exponential_weights / weighted_ols 残差一致）。"""
        sw = float(np.sum(w))
        if sw <= 0:
            return float('nan')
        m = float(np.sum(w * y) / sw)
        v = float(np.sum(w * (y - m) ** 2) / sw)
        return float(np.sqrt(max(v, 0.0)))

    def _compute_hsigma(self) -> pd.DataFrame:
        daily = self.io.read_metric('收盘价')
        daily = self.io.apply_missing_rules(daily, '收盘价')
        daily = daily.rename(columns={'value': 'price'})
        daily = daily.sort_values(['code', 'date'])
        mret = self.io.load_benchmark_returns()
        rows: list[tuple] = []
        for (code, g) in daily.groupby('code'):
            g = g.sort_values('date')
            g = g.merge(mret.rename('mret').reset_index(), on='date', how='inner')
            g['ret'] = pd.to_numeric(g['price'], errors='coerce').pct_change()
            g = g.dropna(subset=['ret'])
            if len(g) < HBETA_WINDOW + 2:
                continue
            rets = g['ret'].values.astype(float)
            mx = g['mret'].values.astype(float)
            dates = g['date'].values
            n = len(g)
            for i in range(HBETA_WINDOW - 1, n):
                y = rets[i - HBETA_WINDOW + 1:i + 1]
                x = mx[i - HBETA_WINDOW + 1:i + 1]
                if not (np.all(np.isfinite(y)) and np.all(np.isfinite(x))):
                    continue
                (a, b) = FactorPipeline._weighted_ols_intercept_slope(y, x, self.W_HBETA)
                resid = y - a - b * x
                hs = self._weighted_std(resid, self.W_HBETA)
                rows.append((pd.Timestamp(dates[i]), code, hs))
        if not rows:
            return pd.DataFrame(columns=['date', 'code', 'HSIGMA'])
        return pd.DataFrame(rows, columns=['date', 'code', 'HSIGMA']).sort_values(['date', 'code']).reset_index(drop=True)

    def _compute_dastd(self) -> pd.DataFrame:
        daily = self.io.read_metric('收盘价')
        daily = self.io.apply_missing_rules(daily, '收盘价')
        daily = daily.rename(columns={'value': 'price'})
        daily = daily.sort_values(['code', 'date'])
        rows = []
        for (code, g) in daily.groupby('code'):
            g = g.sort_values('date')
            r = pd.to_numeric(g['price'], errors='coerce').pct_change().values.astype(float)
            dates = g['date'].values
            for i in range(len(g)):
                if i < DASTD_WINDOW - 1:
                    rows.append((pd.Timestamp(dates[i]), code, np.nan))
                    continue
                seg = r[i - DASTD_WINDOW + 1:i + 1]
                if not np.all(np.isfinite(seg)):
                    dstd = np.nan
                else:
                    dstd = self._weighted_std(seg, self.W_DASTD)
                rows.append((pd.Timestamp(dates[i]), code, dstd))
        return pd.DataFrame(rows, columns=['date', 'code', 'DASTD']).sort_values(['date', 'code']).reset_index(drop=True)

    def _compute_cmra(self) -> pd.DataFrame:
        daily = self.io.read_metric('收盘价')
        daily = self.io.apply_missing_rules(daily, '收盘价')
        daily = daily.rename(columns={'value': 'price'})
        daily = daily.sort_values(['code', 'date'])
        daily['price'] = pd.to_numeric(daily['price'], errors='coerce')
        daily['ym'] = pd.to_datetime(daily['date']).dt.to_period('M')
        month_end = daily.groupby(['code', 'ym'], as_index=False).last()
        month_end = month_end.sort_values(['code', 'ym'])
        month_end['mret'] = month_end.groupby('code')['price'].pct_change()
        month_end['mlr'] = np.log1p(month_end['mret'].clip(lower=-0.9999))
        rows_m: list[tuple] = []
        for (code, g) in month_end.groupby('code'):
            g = g.sort_values('ym')
            mlr = g['mlr'].values.astype(float)
            dts = g['date'].values
            for i in range(len(g)):
                if i < 11:
                    continue
                seg = mlr[i - 11:i + 1]
                if not np.all(np.isfinite(seg)):
                    continue
                vals = [float(np.sum(seg[-T:])) for T in range(1, 13)]
                cmra = float(np.max(vals) - np.min(vals))
                rows_m.append((pd.Timestamp(dts[i]), code, cmra))
        if not rows_m:
            return pd.DataFrame(columns=['date', 'code', 'CMRA'])
        m_df = pd.DataFrame(rows_m, columns=['date', 'code', 'CMRA']).sort_values(['date', 'code'])
        daily_grid = daily[['date', 'code']].drop_duplicates().sort_values(['date', 'code'])
        out = pd.merge_asof(daily_grid, m_df, on='date', by='code', direction='backward')
        return out.sort_values(['date', 'code']).reset_index(drop=True)

    def _assemble_descriptors(self, kind: str, descriptors: list[str]) -> pd.DataFrame:
        """DASTD / HSIGMA / CMRA 均已按附录近似实现。"""
        need = set(self._descriptor_cols) if kind == 'level1' else set(descriptors)
        if not need:
            return pd.DataFrame(columns=['date', 'code'] + self._descriptor_cols)
        dfs: list[pd.DataFrame] = []
        if 'DASTD' in need:
            dfs.append(self._compute_dastd())
        if 'HSIGMA' in need:
            dfs.append(self._compute_hsigma())
        if 'CMRA' in need:
            dfs.append(self._compute_cmra())
        if not dfs:
            return pd.DataFrame(columns=['date', 'code'] + self._descriptor_cols)
        out = dfs[0]
        for d in dfs[1:]:
            out = out.merge(d, on=['date', 'code'], how='outer')
        cols = ['date', 'code'] + [c for c in self._descriptor_cols if c in need]
        return out[cols].sort_values(['date', 'code']).reset_index(drop=True)

    def _compute_level1(self, df: pd.DataFrame) -> pd.DataFrame:
        return FactorPipeline._merge_descriptors_to_level1(df, self._descriptor_cols, self.name, self.name, date_col='date')

class MidCapitalizationFactor(BarraFactor):
    """基于规模因子计算非线性的中等市值因子。"""

    name = 'MidCapitalization'
    descriptors = ('MIDCAP',)
    output_file = 'style_midcap_long.csv'
    _descriptor_cols = ['MIDCAP']
    SIZE_CSV = DATA_FACTORS / 'style_size_long.csv'

    def _read_lncap_from_size_csv(self) -> pd.DataFrame:
        df = read_csv_with_encoding_fallback(self.SIZE_CSV)
        df['date'] = pd.to_datetime(df['date'])
        df['code'] = df['code'].astype(str).str.strip()
        return df[['date', 'code', 'LNCAP']].dropna(subset=['LNCAP'])

    def _compute_midcap(self) -> pd.DataFrame:
        base = self._read_lncap_from_size_csv()
        if base.empty:
            return pd.DataFrame(columns=['date', 'code', 'MIDCAP'])
        base['lncap_cube'] = base['LNCAP'] ** 3
        mids: list[pd.DataFrame] = []
        for (_, sub) in base.groupby('date'):
            y = sub['lncap_cube'].values.astype(float)
            x = sub['LNCAP'].values.astype(float)
            resid = FactorPipeline._cross_section_residual(y, x)
            tmp = sub.copy()
            tmp['_resid'] = resid
            mids.append(tmp)
        base = pd.concat(mids, ignore_index=True)
        wins_parts: list[pd.DataFrame] = []
        for (_, g) in base.groupby('date', sort=False):
            g = g.copy()
            g['_wins'] = FactorPipeline._winsor_series_mad(g['_resid'], n=3.0)
            wins_parts.append(g)
        base = pd.concat(wins_parts, ignore_index=True)
        z_parts: list[pd.DataFrame] = []
        for (_, g) in base.groupby('date', sort=False):
            g = g.copy()
            m = float(g['_wins'].mean())
            sd = float(g['_wins'].std(ddof=0))
            if sd == 0.0 or np.isnan(sd):
                g['MIDCAP'] = np.nan
            else:
                g['MIDCAP'] = (g['_wins'] - m) / sd
            z_parts.append(g)
        base = pd.concat(z_parts, ignore_index=True)
        return base[['date', 'code', 'MIDCAP']].sort_values(['date', 'code']).reset_index(drop=True)

    def _compute_level1(self, df: pd.DataFrame) -> pd.DataFrame:
        return FactorPipeline._merge_descriptors_to_level1(df, self._descriptor_cols, self.name, self.name, date_col='date')

    def _assemble_descriptors(self, kind: str, descriptors: list[str]) -> pd.DataFrame:
        need = set(self._descriptor_cols) if kind == 'level1' else set(descriptors)
        if 'MIDCAP' in need:
            return self._compute_midcap()
        return pd.DataFrame(columns=['date', 'code'] + self._descriptor_cols)

class ProfitabilityFactor(BarraFactor):
    """计算盈利能力因子及资产周转、利润率和资产收益描述子。"""

    name = 'Profitability'
    descriptors = ('ATO', 'GP', 'GPM', 'ROA')
    output_file = 'style_profitability_long.csv'
    _descriptor_cols = ['ATO', 'GP', 'GPM', 'ROA']
    _METRIC_RENAME = {'总资产': 'TA', '营业收入_TTM': 'Sales_TTM', '净利润_TTM': 'NI_TTM', '营业收入_年报': 'Sales_Annual', '营业成本_年报': 'COGS_Annual'}

    def _merge_quarterly_inputs(self, metric_names: list[str]) -> pd.DataFrame | None:
        """按 report_date×code 内连接。"""
        if not metric_names:
            return None
        df: pd.DataFrame | None = None
        for logical in metric_names:
            raw = self.io.read_optional_metric(logical)
            if raw is None:
                return None
            col = self._METRIC_RENAME[logical]
            part = raw.rename(columns={'value': col})
            if df is None:
                df = part
            else:
                df = df.merge(part, on=['report_date', 'code'], how='inner')
        return df

    def _outer_combine(self, parts: list[pd.DataFrame]) -> pd.DataFrame:
        """多个仅含部分 Descriptor 列的季频表，按 report_date×code 外连接。"""
        parts = [p for p in parts if p is not None and (not p.empty)]
        if not parts:
            return pd.DataFrame()
        out = parts[0]
        for p in parts[1:]:
            out = out.merge(p, on=['report_date', 'code'], how='outer')
        return out

    def _compute_from_quarterly(self, need: set[str]) -> pd.DataFrame | None:
        """TTM 类（ATO、ROA）与年报类（GP、GPM）分别合并，再外连成一张季频表后贴日。"""
        if not need:
            return None
        pieces: list[pd.DataFrame] = []
        if need & {'ATO', 'ROA'}:
            ttm_metrics: list[str] = ['总资产']
            if 'ATO' in need:
                ttm_metrics.append('营业收入_TTM')
            if 'ROA' in need:
                ttm_metrics.append('净利润_TTM')
            df_t = self._merge_quarterly_inputs(ttm_metrics)
            if df_t is not None and (not df_t.empty):
                row = df_t.copy()
                if 'ATO' in need:
                    row['ATO'] = np.where(row['TA'].notna() & row['Sales_TTM'].notna() & (row['TA'] > 0) & (row['Sales_TTM'] > 0), row['Sales_TTM'] / row['TA'], np.nan)
                if 'ROA' in need:
                    row['ROA'] = np.where(row['TA'].notna() & row['NI_TTM'].notna() & (row['TA'] > 0), row['NI_TTM'] / row['TA'], np.nan)
                keep = ['report_date', 'code'] + [c for c in ('ATO', 'ROA') if c in need]
                pieces.append(row[keep])
        if need & {'GP', 'GPM'}:
            if 'GP' in need:
                ann_metrics = ['总资产', '营业收入_年报', '营业成本_年报']
            else:
                ann_metrics = ['营业收入_年报', '营业成本_年报']
            df_a = self._merge_quarterly_inputs(ann_metrics)
            if df_a is not None and (not df_a.empty):
                row = df_a.copy()
                gp_num = row['Sales_Annual'] - row['COGS_Annual']
                if 'GP' in need:
                    row['GP'] = np.where(row['TA'].notna() & (row['TA'] > 0), gp_num / row['TA'], np.nan)
                if 'GPM' in need:
                    row['GPM'] = np.where(row['Sales_Annual'].notna() & (row['Sales_Annual'] > 0), gp_num / row['Sales_Annual'], np.nan)
                keep = ['report_date', 'code'] + [c for c in ('GP', 'GPM') if c in need]
                pieces.append(row[keep])
        combined = self._outer_combine(pieces)
        if combined.empty:
            return None
        value_cols = [c for c in self._descriptor_cols if c in need and c in combined.columns]
        if not value_cols:
            return None
        daily_grid = self.io.get_daily_date_code_grid()
        return self.io.quarterly_values_to_daily(combined, value_cols, daily_grid)

    def _assemble_descriptors(self, kind: str, descriptors: list[str]) -> pd.DataFrame:
        need = set(self._descriptor_cols) if kind == 'level1' else set(descriptors)
        cols = [c for c in self._descriptor_cols if c in need]
        if not cols:
            return pd.DataFrame(columns=['date', 'code'] + self._descriptor_cols)
        raw = self._compute_from_quarterly(need)
        if raw is None or raw.empty:
            return self.io.nan_descriptors_from_daily(cols)
        return raw[['date', 'code'] + cols].sort_values(['date', 'code']).reset_index(drop=True)

    def _compute_level1(self, df: pd.DataFrame) -> pd.DataFrame:
        return FactorPipeline._merge_descriptors_to_level1(df, self._descriptor_cols, self.name, self.name, date_col='date')

    def _metrics_to_load(self, need: set[str]) -> list[str]:
        """日志用：去重后的逻辑名列表。"""
        s: set[str] = set()
        if need & {'ATO', 'ROA', 'GP'}:
            s.add('总资产')
        if 'ATO' in need:
            s.add('营业收入_TTM')
        if 'ROA' in need:
            s.add('净利润_TTM')
        if need & {'GP', 'GPM'}:
            s.add('营业收入_年报')
            s.add('营业成本_年报')
        order = ['总资产', '营业收入_TTM', '净利润_TTM', '营业收入_年报', '营业成本_年报']
        return [x for x in order if x in s]

    def _metric_source_files_for_print(self, need: set[str]) -> str:
        logical = self._metrics_to_load(need)
        parts = ['收盘价'] + logical + ['实际披露日期', '预计披露日期']
        return self.io.metric_source_files_str(*parts)

class LeverageFactor(BarraFactor):
    """计算杠杆因子及市场、账面和债务权益描述子。"""

    name = 'Leverage'
    descriptors = ('MLEV', 'BLEV', 'DTOA')
    output_file = 'style_leverage_long.csv'
    _descriptor_cols = ['MLEV', 'BLEV', 'DTOA']

    def _dtoa_daily(self) -> pd.DataFrame | None:
        ta = self.io.read_optional_metric('总资产_年报')
        tl = self.io.read_optional_metric('总负债_年报')
        if ta is None or tl is None:
            return None
        df = ta.rename(columns={'value': 'TA'}).merge(tl.rename(columns={'value': 'TL'}), on=['report_date', 'code'], how='inner')
        df = df.dropna(subset=['TA', 'TL'])
        df = df[df['TA'] > 0]
        df['DTOA'] = df['TL'] / df['TA']
        return self.io.quarterly_values_to_daily(df, ['DTOA'], self.io.get_daily_date_code_grid())

    def _mlev_daily(self) -> pd.DataFrame | None:
        pe = self.io.read_optional_metric('优先股_年报')
        ld = self.io.read_optional_metric('长期负债_年报')
        if pe is None or ld is None:
            return None
        df = pe.rename(columns={'value': 'PE'}).merge(ld.rename(columns={'value': 'LD'}), on=['report_date', 'code'], how='outer')
        df['PE'] = pd.to_numeric(df['PE'], errors='coerce').fillna(0.0)
        df['LD'] = pd.to_numeric(df['LD'], errors='coerce').fillna(0.0)
        daily = self.io.quarterly_values_to_daily(df, ['PE', 'LD'], self.io.get_daily_date_code_grid())
        out = self.io.lag_mkt_cap_one_trading_day(self.io.merge_mkt_cap(daily))
        out['MLEV'] = (out['mkt_cap'] + out['PE'] + out['LD']) / out['mkt_cap']
        return out[['date', 'code', 'MLEV']]

    def _blev_daily(self) -> pd.DataFrame | None:
        be = self.io.read_optional_metric('股东权益_年报')
        pe = self.io.read_optional_metric('优先股_年报')
        ld = self.io.read_optional_metric('长期负债_年报')
        if be is None or pe is None or ld is None:
            return None
        df = be.rename(columns={'value': 'BE'}).merge(pe.rename(columns={'value': 'PE'}), on=['report_date', 'code'], how='outer').merge(ld.rename(columns={'value': 'LD'}), on=['report_date', 'code'], how='outer')
        for c in ('BE', 'PE', 'LD'):
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
        daily = self.io.quarterly_values_to_daily(df, ['BE', 'PE', 'LD'], self.io.get_daily_date_code_grid())
        out = self.io.lag_mkt_cap_one_trading_day(self.io.merge_mkt_cap(daily))
        out['BLEV'] = (out['BE'] + out['PE'] + out['LD']) / out['mkt_cap']
        return out[['date', 'code', 'BLEV']]

    def _merge_outer(self, parts: list[pd.DataFrame]) -> pd.DataFrame:
        if not parts:
            return pd.DataFrame(columns=['date', 'code'] + self._descriptor_cols)
        out = parts[0]
        for p in parts[1:]:
            out = out.merge(p, on=['date', 'code'], how='outer')
        return out.sort_values(['date', 'code']).reset_index(drop=True)

    def _compute_leverage_descriptors(self, need: set[str] | None=None) -> pd.DataFrame:
        if need is None:
            need = set(self._descriptor_cols)
        parts = []
        if 'DTOA' in need:
            d = self._dtoa_daily()
            if d is not None and (not d.empty):
                parts.append(d[['date', 'code', 'DTOA']])
        if 'MLEV' in need:
            m = self._mlev_daily()
            if m is not None and (not m.empty):
                parts.append(m)
        if 'BLEV' in need:
            b = self._blev_daily()
            if b is not None and (not b.empty):
                parts.append(b)
        if not parts:
            return self.io.nan_descriptors_from_daily([c for c in self._descriptor_cols if c in need])
        out = self._merge_outer(parts)
        cols = [c for c in self._descriptor_cols if c in need]
        for c in cols:
            if c not in out.columns:
                out[c] = np.nan
        return out[['date', 'code'] + cols]

    def _assemble_descriptors(self, kind: str, descriptors: list[str]) -> pd.DataFrame:
        need = set(self._descriptor_cols) if kind == 'level1' else set(descriptors)
        cols = [c for c in self._descriptor_cols if c in need]
        if not cols:
            return pd.DataFrame(columns=['date', 'code'] + self._descriptor_cols)
        full = self._compute_leverage_descriptors(need)
        if full.empty:
            return self.io.nan_descriptors_from_daily(cols)
        return full[['date', 'code'] + cols]

    def _compute_level1(self, df: pd.DataFrame) -> pd.DataFrame:
        return FactorPipeline._merge_descriptors_to_level1(df, self._descriptor_cols, self.name, self.name, date_col='date')

    def _leverage_metric_files_for_print(self, need: set[str]) -> str:
        """日志：按运行目标列出会去读的指标逻辑名（去重后交 metric_source_files_str）。"""
        parts: list[str] = []
        if need & {'DTOA'}:
            parts.extend(['总资产_年报', '总负债_年报'])
        if need & {'MLEV', 'BLEV'}:
            parts.extend(['优先股_年报', '长期负债_年报', '总市值'])
        if need & {'BLEV'}:
            parts.append('股东权益_年报')
        parts.extend(['实际披露日期', '预计披露日期', '收盘价'])
        seen: list[str] = []
        for p in parts:
            if p not in seen:
                seen.append(p)
        return self.io.metric_source_files_str(*seen)

class EarningsQualityFactor(BarraFactor):
    """计算盈利质量因子及现金流、应计和利润质量描述子。"""

    name = 'EarningsQuality'
    descriptors = ('ABS', 'ACF')
    output_file = 'style_earnings_quality_long.csv'
    _descriptor_cols = ['ABS', 'ACF']
    _METRIC_KEYS: tuple[str, ...] = ('净利润_单期', '经营活动现金流', '投资活动现金流', '折旧摊销', '总资产', '总负债', '现金及现金等价物', '带息债务')

    def _acf_quarterly(self) -> pd.DataFrame | None:
        ni = self.io.read_optional_metric('净利润_单期')
        cfo = self.io.read_optional_metric('经营活动现金流')
        cfi = self.io.read_optional_metric('投资活动现金流')
        da = self.io.read_optional_metric('折旧摊销')
        ta = self.io.read_optional_metric('总资产')
        if any((x is None for x in (ni, cfo, cfi, da, ta))):
            return None
        df = ni.rename(columns={'value': 'NI'}).merge(cfo.rename(columns={'value': 'CFO'}), on=['report_date', 'code'], how='inner').merge(cfi.rename(columns={'value': 'CFI'}), on=['report_date', 'code'], how='inner').merge(da.rename(columns={'value': 'DA'}), on=['report_date', 'code'], how='inner').merge(ta.rename(columns={'value': 'TA'}), on=['report_date', 'code'], how='inner')
        df['ACCR_CF'] = df['NI'] - df['CFO'] - df['CFI'] + df['DA']
        df['ACF'] = -df['ACCR_CF'] / df['TA'].where(df['TA'] > 0, np.nan)
        return self.io.quarterly_values_to_daily(df, ['ACF'], self.io.get_daily_date_code_grid())

    def _abs_quarterly(self) -> pd.DataFrame | None:
        ta = self.io.read_optional_metric('总资产')
        tl = self.io.read_optional_metric('总负债')
        cash = self.io.read_optional_metric('现金及现金等价物')
        td = self.io.read_optional_metric('带息债务')
        da = self.io.read_optional_metric('折旧摊销')
        if any((x is None for x in (ta, tl, cash, td, da))):
            return None
        df = ta.rename(columns={'value': 'TA'}).merge(tl.rename(columns={'value': 'TL'}), on=['report_date', 'code'], how='inner').merge(cash.rename(columns={'value': 'Cash'}), on=['report_date', 'code'], how='inner').merge(td.rename(columns={'value': 'TD'}), on=['report_date', 'code'], how='inner').merge(da.rename(columns={'value': 'DA'}), on=['report_date', 'code'], how='inner')
        df['NOA'] = df['TA'] - df['Cash'] - (df['TL'] - df['TD'])
        df = df.sort_values(['code', 'report_date'])
        df['NOA_lag'] = df.groupby('code')['NOA'].shift(1)
        df['ACCR_BS'] = df['NOA'] - df['NOA_lag'] - df['DA']
        df['ABS'] = -df['ACCR_BS'] / df['TA'].where(df['TA'] > 0, np.nan)
        return self.io.quarterly_values_to_daily(df, ['ABS'], self.io.get_daily_date_code_grid())

    def _assemble_descriptors(self, kind: str, descriptors: list[str]) -> pd.DataFrame:
        """按运行目标只计算需要的二级 Descriptor，避免每次算满 ABS+ACF。"""
        need = set(self._descriptor_cols) if kind == 'level1' else set(descriptors)
        cols = [c for c in self._descriptor_cols if c in need]
        if not cols:
            return pd.DataFrame(columns=['date', 'code'] + self._descriptor_cols)
        parts: list[pd.DataFrame] = []
        if 'ACF' in need:
            a = self._acf_quarterly()
            if a is not None and (not a.empty):
                parts.append(a[['date', 'code', 'ACF']])
        if 'ABS' in need:
            b = self._abs_quarterly()
            if b is not None and (not b.empty):
                parts.append(b[['date', 'code', 'ABS']])
        if not parts:
            return self.io.nan_descriptors_from_daily(cols)
        out = parts[0]
        for p in parts[1:]:
            out = out.merge(p, on=['date', 'code'], how='outer')
        for c in cols:
            if c not in out.columns:
                out[c] = np.nan
        return out[['date', 'code'] + cols]

    def _compute_level1(self, df: pd.DataFrame) -> pd.DataFrame:
        return FactorPipeline._merge_descriptors_to_level1(df, self._descriptor_cols, self.name, self.name, date_col='date')

class InvestmentQualityFactor(BarraFactor):
    """计算投资质量因子及资产、资本开支和存货增长描述子。"""

    name = 'InvestmentQuality'
    descriptors = ('AGRO', 'IGRO', 'CXGRO')
    output_file = 'style_investment_quality_long.csv'
    _descriptor_cols = ['AGRO', 'IGRO', 'CXGRO']
    _FISCAL_PERIODS = 5
    _FISCAL_MIN_OBS = 5
    _DESCRIPTOR_TO_METRIC: dict[str, str] = {'AGRO': '总资产_年报', 'IGRO': '流通股本_年报', 'CXGRO': '资本支出_年报'}

    def _ordered_descriptors_to_compute(self, kind: str, descriptors: list[str]) -> list[str]:
        """本次运行需要计算的二级因子（有序，与 ``DESCRIPTOR_COLS`` 一致）。"""
        need = set(self._descriptor_cols) if kind == 'level1' else set(descriptors)
        return [c for c in self._descriptor_cols if c in need]

    def _bohai_slope_over_mean_then_negate(self, metric_logical_name: str, descriptor_code: str) -> pd.DataFrame | None:
        """
    单指标：渤海「最近/过去 5 个财政年度」斜率÷|均值|，再取相反数；输出日频单列 ``descriptor_code``。
    仅在本函数内 ``read_optional_metric``，避免为未选中的因子读盘。
    """
        raw = self.io.read_optional_metric(metric_logical_name)
        if raw is None:
            return None
        df = raw.rename(columns={'value': 'v'}).sort_values(['code', 'report_date'])
        rows: list[tuple] = []
        for (code, g) in df.groupby('code', sort=False):
            g = g.sort_values('report_date')
            for i in range(len(g)):
                sub = g['v'].iloc[:i + 1]
                rd = g['report_date'].iloc[i]
                if len(sub) < self._FISCAL_MIN_OBS:
                    rows.append((rd, code, np.nan))
                else:
                    r = FactorPipeline._five_year_regression_scaled(sub, periods=self._FISCAL_PERIODS, min_points=self._FISCAL_MIN_OBS)
                    rows.append((rd, code, -r if np.isfinite(r) else np.nan))
        if not rows:
            return None
        q = pd.DataFrame(rows, columns=['report_date', 'code', descriptor_code])
        return self.io.quarterly_values_to_daily(q, [descriptor_code], self.io.get_daily_date_code_grid())

    def _assemble_descriptors(self, kind: str, descriptors: list[str]) -> pd.DataFrame:
        """按 ``kind`` / ``descriptors`` 仅计算需要的二级因子并合并。"""
        cols = self._ordered_descriptors_to_compute(kind, descriptors)
        if not cols:
            return pd.DataFrame(columns=['date', 'code'] + self._descriptor_cols)
        parts: list[pd.DataFrame] = []
        for code in cols:
            metric = self._DESCRIPTOR_TO_METRIC[code]
            block = self._bohai_slope_over_mean_then_negate(metric, code)
            if block is not None and (not block.empty):
                parts.append(block)
        if not parts:
            return self.io.nan_descriptors_from_daily(cols)
        out = parts[0]
        for p in parts[1:]:
            out = out.merge(p, on=['date', 'code'], how='outer')
        for c in cols:
            if c not in out.columns:
                out[c] = np.nan
        return out[['date', 'code'] + cols]

    def _compute_level1(self, df: pd.DataFrame) -> pd.DataFrame:
        return FactorPipeline._merge_descriptors_to_level1(df, self._descriptor_cols, self.name, self.name, date_col='date')

    def _metric_keys_for_run(self, kind: str, descriptors: list[str]) -> tuple[str, ...]:
        """本次实际会用到的财报逻辑名（去重、顺序稳定），用于日志。"""
        cols = self._ordered_descriptors_to_compute(kind, descriptors)
        return tuple(dict.fromkeys((self._DESCRIPTOR_TO_METRIC[c] for c in cols)))

class GrowthFactor(BarraFactor):
    """计算成长因子及长期盈利、盈利和收入增长描述子。"""

    name = 'Growth'
    descriptors = ('EGRLF', 'EGRO', 'SGRO')
    output_file = 'style_growth_long.csv'
    _descriptor_cols = ['EGRLF', 'EGRO', 'SGRO']
    _FISCAL_PERIODS = 5
    _FISCAL_MIN_OBS = 5
    EGRLF_FY3_HORIZON_YEARS = 3

    def _egrlf_daily(self) -> pd.DataFrame | None:
        """FY3 与年报净利润联合得到的长期盈利增速代理：CAGR ≈ (FY3/NI)^(1/3)-1。"""
        fc = self.io.read_optional_metric('一致预期净利润_FY3')
        ni = self.io.read_optional_metric('净利润_年报')
        if fc is None or ni is None:
            return None
        fc = fc.rename(columns={'value': 'fc_fy3'})
        ni = ni.rename(columns={'value': 'ni'})
        df = fc.merge(ni, on=['report_date', 'code'], how='inner')
        df = df.dropna(subset=['fc_fy3', 'ni'])
        ok = (df['ni'] > 0) & (df['fc_fy3'] > 0)
        if not ok.any():
            return None
        h = float(self.EGRLF_FY3_HORIZON_YEARS)
        df['EGRLF'] = np.nan
        df.loc[ok, 'EGRLF'] = (df.loc[ok, 'fc_fy3'] / df.loc[ok, 'ni']).pow(1.0 / h) - 1.0
        df = df.dropna(subset=['EGRLF'])
        if df.empty:
            return None
        daily = self.io.quarterly_values_to_daily(df, ['EGRLF'], self.io.get_daily_date_code_grid())
        return daily[['date', 'code', 'EGRLF']]

    def _egro_sgro_daily(self, metric_logical_name: str, out_col: str) -> pd.DataFrame | None:
        """渤海附录：5 个财政年度每股序列对时间回归斜率 ÷ |均值|（与 AGRO 同构，不取相反数）。"""
        raw = self.io.read_optional_metric(metric_logical_name)
        if raw is None:
            return None
        df = raw.rename(columns={'value': 'v'}).sort_values(['code', 'report_date'])
        rows: list[tuple] = []
        for (code, g) in df.groupby('code', sort=False):
            g = g.sort_values('report_date')
            for i in range(len(g)):
                sub = g['v'].iloc[:i + 1]
                rd = g['report_date'].iloc[i]
                if len(sub) < self._FISCAL_MIN_OBS:
                    rows.append((rd, code, np.nan))
                else:
                    r = FactorPipeline._five_year_regression_scaled(sub, periods=self._FISCAL_PERIODS, min_points=self._FISCAL_MIN_OBS)
                    rows.append((rd, code, r if np.isfinite(r) else np.nan))
        if not rows:
            return None
        q = pd.DataFrame(rows, columns=['report_date', 'code', out_col])
        if q.empty:
            return None
        daily = self.io.quarterly_values_to_daily(q, [out_col], self.io.get_daily_date_code_grid())
        return daily[['date', 'code', out_col]]

    def _egro_daily(self) -> pd.DataFrame | None:
        return self._egro_sgro_daily('每股收益', 'EGRO')

    def _sgro_daily(self) -> pd.DataFrame | None:
        return self._egro_sgro_daily('每股营业收入', 'SGRO')

    def _merge_outer(self, parts: list[pd.DataFrame]) -> pd.DataFrame:
        if not parts:
            return pd.DataFrame(columns=['date', 'code'])
        out = parts[0]
        for p in parts[1:]:
            out = out.merge(p, on=['date', 'code'], how='outer')
        return out.sort_values(['date', 'code']).reset_index(drop=True)

    def _ordered_descriptors_to_compute(self, kind: str, descriptors: list[str]) -> list[str]:
        """本次运行需要计算的二级因子（顺序与 ``DESCRIPTOR_COLS`` 一致）。"""
        need = set(self._descriptor_cols) if kind == 'level1' else set(descriptors)
        return [c for c in self._descriptor_cols if c in need]

    def _compute_selected_descriptors(self, cols: list[str]) -> pd.DataFrame:
        """仅计算 ``cols`` 中列出的描述子，避免未选中的因子仍读盘、仍回归。"""
        if not cols:
            return pd.DataFrame(columns=['date', 'code'])
        parts: list[pd.DataFrame] = []
        if 'EGRLF' in cols:
            d = self._egrlf_daily()
            if d is not None and (not d.empty):
                parts.append(d)
        if 'EGRO' in cols:
            d = self._egro_daily()
            if d is not None and (not d.empty):
                parts.append(d)
        if 'SGRO' in cols:
            d = self._sgro_daily()
            if d is not None and (not d.empty):
                parts.append(d)
        if not parts:
            return pd.DataFrame(columns=['date', 'code'])
        return self._merge_outer(parts)

    def _metric_logical_names_for_run(self, kind: str, descriptors: list[str]) -> tuple[str, ...]:
        """本次实际会读到的 ``METRIC_FILES`` 逻辑名（去重、顺序稳定），用于日志。"""
        cols = self._ordered_descriptors_to_compute(kind, descriptors)
        keys: list[str] = ['收盘价']
        if 'EGRLF' in cols:
            keys.extend(['一致预期净利润_FY3', '净利润_年报'])
        if 'EGRO' in cols:
            keys.append('每股收益')
        if 'SGRO' in cols:
            keys.append('每股营业收入')
        return tuple(dict.fromkeys(keys))

    def _assemble_descriptors(self, kind: str, descriptors: list[str]) -> pd.DataFrame:
        cols = self._ordered_descriptors_to_compute(kind, descriptors)
        if not cols:
            return pd.DataFrame(columns=['date', 'code'] + self._descriptor_cols)
        full = self._compute_selected_descriptors(cols)
        if full.empty or len(full.columns) <= 2:
            return self.io.nan_descriptors_from_daily(cols)
        return full[['date', 'code'] + cols]

    def _compute_level1(self, df: pd.DataFrame) -> pd.DataFrame:
        return FactorPipeline._merge_descriptors_to_level1(df, self._descriptor_cols, self.name, self.name, date_col='date')

class EarningsVariabilityFactor(BarraFactor):
    """计算盈利波动因子及收入、盈利、现金流和预期波动描述子。"""

    name = 'EarningsVariability'
    descriptors = ('VSAL', 'VERN', 'VFLO', 'ETOPF_STD')
    output_file = 'style_earnings_variability_long.csv'
    _descriptor_cols = ['VSAL', 'VERN', 'VFLO', 'ETOPF_STD']
    _FISCAL_YEARS = 5

    def _var_descriptor_five_fiscal_years(self, metric_name: str, out_col: str) -> pd.DataFrame | None:
        """附录：过去五个财年之年频序列的 std/mean；年报宽表一行一财年。"""
        raw = self.io.read_optional_metric(metric_name)
        if raw is None:
            return None
        df = raw.rename(columns={'value': 'v'}).sort_values(['code', 'report_date'])
        rows: list[tuple] = []
        n = self._FISCAL_YEARS
        for (code, g) in df.groupby('code'):
            g = g.sort_values('report_date').reset_index(drop=True)
            for i in range(len(g)):
                if i < n - 1:
                    r = float('nan')
                else:
                    tail = g['v'].iloc[i - n + 1:i + 1].astype(float).values
                    r = FactorPipeline._rolling_std_mean_ratio(tail)
                rows.append((g['report_date'].iloc[i], code, r))
        if not rows:
            return None
        q = pd.DataFrame(rows, columns=['report_date', 'code', out_col])
        return self.io.quarterly_values_to_daily(q, [out_col], self.io.get_daily_date_code_grid())

    def _etopf_std_daily(self) -> pd.DataFrame | None:
        st = self.io.read_optional_metric('分析师预测EP标准差')
        if st is None:
            return None
        df = st.rename(columns={'value': 'eps_std'})
        daily = self.io.quarterly_values_to_daily(df, ['eps_std'], self.io.get_daily_date_code_grid())
        px = self.io.read_metric('收盘价')
        px = self.io.apply_missing_rules(px, '收盘价').rename(columns={'value': 'price'})
        out = daily.merge(px, on=['date', 'code'], how='inner')
        out = out[out['price'] > 0]
        out['ETOPF_STD'] = out['eps_std'] / out['price']
        return out[['date', 'code', 'ETOPF_STD']]

    def _assemble_descriptors(self, kind: str, descriptors: list[str]) -> pd.DataFrame:
        """一级：算齐四个描述子；二级：只算 ``descriptors`` 中出现的列（与 ``compute_dividend_yield`` 同构）。"""
        need = set(self._descriptor_cols) if kind == 'level1' else set(descriptors)
        cols = [c for c in self._descriptor_cols if c in need]
        if not cols:
            return pd.DataFrame(columns=['date', 'code'] + self._descriptor_cols)
        parts: list[pd.DataFrame] = []
        if 'VSAL' in need:
            d = self._var_descriptor_five_fiscal_years('营业收入_年报', 'VSAL')
            if d is not None and (not d.empty):
                parts.append(d)
        if 'VERN' in need:
            d = self._var_descriptor_five_fiscal_years('净利润_年报', 'VERN')
            if d is not None and (not d.empty):
                parts.append(d)
        if 'VFLO' in need:
            d = self._var_descriptor_five_fiscal_years('现金及现金等价物_年报', 'VFLO')
            if d is not None and (not d.empty):
                parts.append(d)
        if 'ETOPF_STD' in need:
            e = self._etopf_std_daily()
            if e is not None and (not e.empty):
                parts.append(e)
        if not parts:
            return self.io.nan_descriptors_from_daily(cols)
        out = parts[0]
        for p in parts[1:]:
            out = out.merge(p, on=['date', 'code'], how='outer')
        for c in cols:
            if c not in out.columns:
                out[c] = np.nan
        return out[['date', 'code'] + cols].sort_values(['date', 'code']).reset_index(drop=True)

    def _compute_level1(self, df: pd.DataFrame) -> pd.DataFrame:
        return FactorPipeline._merge_descriptors_to_level1(df, self._descriptor_cols, self.name, self.name, date_col='date')

class BookToPriceFactor(BarraFactor):
    """计算账面市值比因子及 BTOP 描述子。"""

    name = 'BookToPrice'
    descriptors = ('BTOP',)
    output_file = 'style_book_to_price_long.csv'
    _descriptor_cols = ['BTOP']

    def _compute_btop(self) -> pd.DataFrame:
        eq = self.io.read_optional_metric('股东权益')
        if eq is None:
            return self.io.nan_descriptors_from_daily(['BTOP'])[['date', 'code', 'BTOP']]
        df = eq.rename(columns={'value': 'equity'})
        df = df.dropna(subset=['equity'])
        df = df[df['equity'] > 0]
        daily = self.io.quarterly_values_to_daily(df, ['equity'], self.io.get_daily_date_code_grid())
        out = self.io.merge_mkt_cap(daily)
        out['BTOP'] = out['equity'] / out['mkt_cap']
        return out[['date', 'code', 'BTOP']].sort_values(['date', 'code']).reset_index(drop=True)

    def _assemble_descriptors(self, kind: str, descriptors: list[str]) -> pd.DataFrame:
        need = set(self._descriptor_cols) if kind == 'level1' else set(descriptors)
        if 'BTOP' in need:
            return self._compute_btop()
        return pd.DataFrame(columns=['date', 'code'] + self._descriptor_cols)

    def _compute_level1(self, df: pd.DataFrame) -> pd.DataFrame:
        return FactorPipeline._merge_descriptors_to_level1(df, self._descriptor_cols, self.name, self.name, date_col='date')

class EarningsYieldFactor(BarraFactor):
    """计算盈利收益率因子及现金、盈利和预期收益描述子。"""

    name = 'EarningsYield'
    descriptors = ('CETOP', 'ETOP', 'EM', 'ETOPF')
    output_file = 'style_earnings_yield_long.csv'
    _descriptor_cols = ['CETOP', 'ETOP', 'EM', 'ETOPF']

    def _etop_daily(self) -> pd.DataFrame | None:
        ni = self.io.read_optional_metric('净利润_TTM')
        if ni is None:
            return None
        df = ni.rename(columns={'value': 'NI'})
        df = df.dropna(subset=['NI'])
        daily = self.io.quarterly_values_to_daily(df, ['NI'], self.io.get_daily_date_code_grid())
        out = self.io.merge_mkt_cap(daily)
        out['ETOP'] = out['NI'] / out['mkt_cap']
        return out[['date', 'code', 'ETOP']]

    def _cetop_daily(self) -> pd.DataFrame | None:
        ce = self.io.read_optional_metric('现金盈利_TTM')
        if ce is None:
            return None
        df = ce.rename(columns={'value': 'CE'})
        df = df.dropna(subset=['CE'])
        daily = self.io.quarterly_values_to_daily(df, ['CE'], self.io.get_daily_date_code_grid())
        out = self.io.merge_mkt_cap(daily)
        out['CETOP'] = out['CE'] / out['mkt_cap']
        return out[['date', 'code', 'CETOP']]

    def _em_daily(self) -> pd.DataFrame | None:
        ebit = self.io.read_optional_metric('息税前利润_年报')
        ev = self.io.read_optional_metric('企业价值')
        if ebit is None or ev is None:
            return None
        ebit_d = self.io.quarterly_values_to_daily(ebit.rename(columns={'value': 'EBIT'}), ['EBIT'], self.io.get_daily_date_code_grid())
        ev = ev.rename(columns={'value': 'EV'})
        out = ebit_d.merge(ev, on=['date', 'code'], how='inner')
        out = out[out['EV'] > 0]
        out['EM'] = out['EBIT'] / out['EV']
        return out[['date', 'code', 'EM']]

    def _etopf_daily(self) -> pd.DataFrame | None:
        fc = self.io.read_optional_metric('分析师预测盈利')
        if fc is None:
            return None
        df = fc.rename(columns={'value': 'FPROF'})
        df = df.dropna(subset=['FPROF'])
        daily = self.io.quarterly_values_to_daily(df, ['FPROF'], self.io.get_daily_date_code_grid())
        out = self.io.merge_mkt_cap(daily)
        out['ETOPF'] = out['FPROF'] / out['mkt_cap']
        return out[['date', 'code', 'ETOPF']]

    def _merge_outer(self, parts: list[pd.DataFrame]) -> pd.DataFrame:
        out = parts[0]
        for p in parts[1:]:
            out = out.merge(p, on=['date', 'code'], how='outer')
        return out.sort_values(['date', 'code']).reset_index(drop=True)

    def _assemble_descriptors(self, kind: str, descriptors: list[str]) -> pd.DataFrame:
        """一级：算齐四个描述子；二级：只算 ``descriptors`` 中出现的列（与 ``compute_dividend_yield`` 同构）。"""
        need = set(self._descriptor_cols) if kind == 'level1' else set(descriptors)
        cols = [c for c in self._descriptor_cols if c in need]
        if not cols:
            return pd.DataFrame(columns=['date', 'code'] + self._descriptor_cols)
        parts: list[pd.DataFrame] = []
        if 'CETOP' in need:
            d = self._cetop_daily()
            if d is not None and (not d.empty):
                parts.append(d)
        if 'ETOP' in need:
            d = self._etop_daily()
            if d is not None and (not d.empty):
                parts.append(d)
        if 'EM' in need:
            d = self._em_daily()
            if d is not None and (not d.empty):
                parts.append(d)
        if 'ETOPF' in need:
            d = self._etopf_daily()
            if d is not None and (not d.empty):
                parts.append(d)
        if not parts:
            return self.io.nan_descriptors_from_daily(cols)
        out = self._merge_outer(parts)
        for c in cols:
            if c not in out.columns:
                out[c] = np.nan
        return out[['date', 'code'] + cols]

    def _compute_level1(self, df: pd.DataFrame) -> pd.DataFrame:
        return FactorPipeline._merge_descriptors_to_level1(df, self._descriptor_cols, self.name, self.name, date_col='date')

class DividendYieldFactor(BarraFactor):
    """计算股息率因子及历史、预期股息率描述子。"""

    name = 'DividendYield'
    descriptors = ('DTOP', 'DTOPF')
    output_file = 'style_dividend_yield_long.csv'
    _descriptor_cols = ['DTOP', 'DTOPF']

    def _compute_dtop(self) -> pd.DataFrame:
        price = self.io.read_metric('收盘价')
        price = self.io.apply_missing_rules(price, '收盘价')
        price = price.rename(columns={'value': 'price'})
        div = self.io.read_metric('过去12个月现金股利')
        div = div.rename(columns={'value': 'dividend'})
        div = div.dropna(subset=['dividend'])
        div = div[div['dividend'] >= 0]
        if div.empty:
            return pd.DataFrame(columns=['date', 'code', 'DTOP'])
        try:
            actual = self.io.read_metric('实际披露日期')
            expected = self.io.read_metric('预计披露日期')
        except (KeyError, FileNotFoundError):
            (actual, expected) = (None, None)
        disclose = self.io.build_disclose_date(div, actual, expected)
        div = div.merge(disclose, on=['report_date', 'code'], how='left')
        div['disclose_date'] = pd.to_datetime(div['disclose_date'], errors='coerce')
        div = div.dropna(subset=['disclose_date'])
        daily_grid = self.io.get_daily_date_code_grid()
        div_daily = self.io.lowfreq_to_daily(div, 'disclose_date', 'dividend', daily_grid)
        df = self.io.attach_prior_month_end_close(div_daily, price)
        df = df[df['price_prior_month_end'].notna() & (df['price_prior_month_end'] > 0)]
        df['DTOP'] = df['dividend'] / df['price_prior_month_end']
        return df[['date', 'code', 'DTOP']].sort_values(['date', 'code']).reset_index(drop=True)

    def _compute_dtopf(self) -> pd.DataFrame:
        try:
            pr = self.io.read_metric('收盘价')
            pr = self.io.apply_missing_rules(pr, '收盘价').rename(columns={'value': 'price'})
            fc = self.io.read_metric('预测每股股利')
            fc = fc.rename(columns={'value': 'fc_dps'})
            fc = fc.dropna(subset=['fc_dps'])
            try:
                actual = self.io.read_metric('实际披露日期')
                expected = self.io.read_metric('预计披露日期')
            except (KeyError, FileNotFoundError):
                (actual, expected) = (None, None)
            disclose = self.io.build_disclose_date(fc, actual, expected)
            fc = fc.merge(disclose, on=['report_date', 'code'], how='left')
            fc['disclose_date'] = pd.to_datetime(fc['disclose_date'], errors='coerce')
            fc = fc.dropna(subset=['disclose_date'])
            daily_grid = self.io.get_daily_date_code_grid()
            fc_d = self.io.lowfreq_to_daily(fc, 'disclose_date', 'fc_dps', daily_grid)
            out = pr.merge(fc_d, on=['date', 'code'], how='inner')
            out = out[out['price'] > 0]
            out['DTOPF'] = out['fc_dps'] / out['price']
            return out[['date', 'code', 'DTOPF']].sort_values(['date', 'code']).reset_index(drop=True)
        except (KeyError, FileNotFoundError, ValueError):
            return self.io.nan_descriptors_from_daily(['DTOPF'])[['date', 'code', 'DTOPF']]

    def _assemble_descriptors(self, kind: str, descriptors: list[str]) -> pd.DataFrame:
        """一级：需 DTOP+DTOPF 再合成；二级：只算 `descriptors` 中出现的列（省资源）。"""
        need = set(self._descriptor_cols) if kind == 'level1' else set(descriptors)
        a = self._compute_dtop() if 'DTOP' in need else None
        b = self._compute_dtopf() if 'DTOPF' in need else None
        if a is None and b is None:
            return pd.DataFrame(columns=['date', 'code'] + self._descriptor_cols)
        if a is None:
            assert b is not None
            return b
        if b is None:
            return a
        if a.empty and b.empty:
            return pd.DataFrame(columns=['date', 'code'] + self._descriptor_cols)
        if a.empty:
            return b
        if b.empty:
            return a
        return a.merge(b, on=['date', 'code'], how='outer').sort_values(['date', 'code']).reset_index(drop=True)

    def _compute_level1(self, df: pd.DataFrame) -> pd.DataFrame:
        return FactorPipeline._merge_descriptors_to_level1(df, self._descriptor_cols, self.name, self.name, date_col='date')

class LongTermReversalFactor(BarraFactor):
    """计算长期反转因子及 LTRSTR、LTHALPHA 描述子。"""

    name = 'LongTermReversal'
    descriptors = ('LTRSTR', 'LTHALPHA')
    output_file = 'style_long_term_reversal_long.csv'
    _descriptor_cols = ['LTRSTR', 'LTHALPHA']
    W_LTR = FactorPipeline._exponential_weights(LTRSTR_WINDOW, LTRSTR_HALFLIFE)
    _SMOOTH_WIN = 11

    def _price_with_mret(self) -> pd.DataFrame:
        daily = self.io.read_metric('收盘价')
        daily = self.io.apply_missing_rules(daily, '收盘价')
        daily = daily.rename(columns={'value': 'price'})
        daily = daily.sort_values(['code', 'date'])
        daily['price'] = pd.to_numeric(daily['price'], errors='coerce')
        daily['ret'] = daily.groupby('code')['price'].pct_change()
        daily = daily.dropna(subset=['ret'])
        mret = self.io.load_benchmark_returns()
        daily = daily.merge(mret.rename('mret').reset_index(), on='date', how='inner')
        daily['logret'] = np.log1p(daily['ret'].clip(lower=-0.9999))
        return daily

    def _compute_ltrstr(self) -> pd.DataFrame:
        daily = self._price_with_mret()
        rows: list[tuple] = []
        need = max(LTRSTR_WINDOW, LTRSTR_LAG + self._SMOOTH_WIN + 5)
        for (code, g) in daily.groupby('code'):
            g = g.sort_values('date')
            if len(g) < need:
                continue
            lr = g['logret'].values.astype(float)
            dates = g['date'].values
            n = len(g)
            raw = np.full(n, np.nan)
            for i in range(n):
                if i >= LTRSTR_WINDOW - 1:
                    seg = lr[i - LTRSTR_WINDOW + 1:i + 1]
                    if np.all(np.isfinite(seg)):
                        raw[i] = float(np.sum(self.W_LTR * seg))
            adj = np.full(n, np.nan)
            for i in range(n):
                if i < LTRSTR_LAG + self._SMOOTH_WIN - 1:
                    continue
                win = raw[i - LTRSTR_LAG - (self._SMOOTH_WIN - 1):i - LTRSTR_LAG + 1]
                if np.all(np.isfinite(win)):
                    adj[i] = -float(np.nanmean(win))
            for i in range(n):
                if np.isfinite(adj[i]):
                    rows.append((pd.Timestamp(dates[i]), code, adj[i]))
        if not rows:
            return pd.DataFrame(columns=['date', 'code', 'LTRSTR'])
        return pd.DataFrame(rows, columns=['date', 'code', 'LTRSTR']).sort_values(['date', 'code']).reset_index(drop=True)

    def _compute_lthalpha(self) -> pd.DataFrame:
        daily = self._price_with_mret()
        rows: list[tuple] = []
        need = max(LTRSTR_WINDOW, LTRSTR_LAG + self._SMOOTH_WIN + 5)
        for (code, g) in daily.groupby('code'):
            g = g.sort_values('date')
            if len(g) < need:
                continue
            rets = g['ret'].values.astype(float)
            mx = g['mret'].values.astype(float)
            dates = g['date'].values
            n = len(g)
            raw = np.full(n, np.nan)
            for i in range(LTRSTR_WINDOW - 1, n):
                y = rets[i - LTRSTR_WINDOW + 1:i + 1]
                x = mx[i - LTRSTR_WINDOW + 1:i + 1]
                if np.all(np.isfinite(y)) and np.all(np.isfinite(x)):
                    (a, _) = FactorPipeline._weighted_ols_intercept_slope(y, x, self.W_LTR)
                    raw[i] = a
            adj = np.full(n, np.nan)
            for i in range(n):
                if i < LTRSTR_LAG + self._SMOOTH_WIN - 1:
                    continue
                win = raw[i - LTRSTR_LAG - (self._SMOOTH_WIN - 1):i - LTRSTR_LAG + 1]
                if np.all(np.isfinite(win)):
                    adj[i] = -float(np.nanmean(win))
            for i in range(n):
                if np.isfinite(adj[i]):
                    rows.append((pd.Timestamp(dates[i]), code, adj[i]))
        if not rows:
            return pd.DataFrame(columns=['date', 'code', 'LTHALPHA'])
        return pd.DataFrame(rows, columns=['date', 'code', 'LTHALPHA']).sort_values(['date', 'code']).reset_index(drop=True)

    def _assemble_descriptors(self, kind: str, descriptors: list[str]) -> pd.DataFrame:
        need = set(self._descriptor_cols) if kind == 'level1' else set(descriptors)
        parts: list[pd.DataFrame] = []
        if 'LTRSTR' in need:
            parts.append(self._compute_ltrstr())
        if 'LTHALPHA' in need:
            parts.append(self._compute_lthalpha())
        if not parts:
            return self.io.nan_descriptors_from_daily(self._descriptor_cols)
        out = parts[0]
        for p in parts[1:]:
            out = out.merge(p, on=['date', 'code'], how='outer')
        return out.sort_values(['date', 'code']).reset_index(drop=True)

    def _compute_level1(self, df: pd.DataFrame) -> pd.DataFrame:
        return FactorPipeline._merge_descriptors_to_level1(df, self._descriptor_cols, self.name, self.name, date_col='date')
BARRA_FACTOR_CLASSES = (
    SizeFactor, BetaFactor, LiquidityFactor, MomentumFactor,
    ResidualVolatilityFactor, MidCapitalizationFactor, ProfitabilityFactor,
    LeverageFactor, EarningsQualityFactor, InvestmentQualityFactor, GrowthFactor,
    EarningsVariabilityFactor, BookToPriceFactor, EarningsYieldFactor,
    DividendYieldFactor, LongTermReversalFactor,
)

ALL_FACTOR_CLASSES = BARRA_FACTOR_CLASSES
FACTOR_CLASS_BY_NAME = {cls.name: cls for cls in ALL_FACTOR_CLASSES}
