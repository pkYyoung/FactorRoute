"""
米筐/API 导出宽表 CSV 的读入：统一 `read_metric(逻辑名)`，不区分「日频/季频」。

宽表形态一致：首列为时间、其余列为标的代码；仅根据配置决定长表时间列名为 `date` 或 `report_date`。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from .. import project_config as default_config
from ..project_config import (
        INDEX_COLUMN,
        INDEX_SHEET,
        METRIC_ALIAS,
        METRIC_ASOF_MODE,
        METRIC_FILES,
        METRIC_LOW_FREQ,
        METRIC_USE_REPORT_DATE,
        METRIC_WIDE_DAILY_CATEGORICAL,
        METRIC_WIDE_REPORT_VALUE_AS_TEXT,
        WIDE_CSV_DIR,
)

# 宽表来源多样：按序严格解码；勿将 latin-1 放入主循环（会误吞任意字节）。
# 中文 GBK CSV 若先于 gbk 尝试 utf-16-be，可能「无解码报错但解析成空表」，故 gbk/gb18030 紧接 UTF-8。
_CSV_ENCODINGS_TRY: tuple[str, ...] = (
    "utf-8-sig",
    "utf-8",
    "gbk",
    "gb18030",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
    "cp1252",
)


def read_csv_with_encoding_fallback(path: Path | str, **kwargs) -> pd.DataFrame:
    """读取 CSV：多编码尝试；仍失败时用 UTF-8 替换非法字节，最后 latin-1 兜底。"""
    p = Path(path)
    last_err: Optional[BaseException] = None
    for enc in _CSV_ENCODINGS_TRY:
        try:
            df = pd.read_csv(p, encoding=enc, **kwargs)
            if enc.startswith("utf-16") and len(df.columns) < 2 and len(df) == 0:
                continue
            return df
        except UnicodeDecodeError as e:
            last_err = e
            continue
    # pandas>=1.3：替换无法解码的字节（混有脏数据时仍能读）
    try:
        return pd.read_csv(
            p, encoding="utf-8", encoding_errors="replace", **kwargs
        )
    except (UnicodeDecodeError, TypeError, ValueError) as e:
        last_err = e
    try:
        return pd.read_csv(p, encoding="latin-1", **kwargs)
    except Exception as e:
        last_err = e
    if last_err is not None:
        raise OSError(f"无法解码 CSV: {p}") from last_err
    raise OSError(f"无法解码 CSV: {p}")


# 旧版 Excel .xls（OLE）文件魔数；扩展名常被误写为 .csv
_OLE_COMPOUND_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def read_wide_metric_file(path: Path | str, *, header: int = 0, low_memory: bool = False) -> pd.DataFrame:
    """读取宽表：支持真 CSV（多编码）及误命名为 .csv 的 Excel（.xls / .xlsx）。"""
    p = Path(path)
    with p.open("rb") as f:
        head = f.read(8)
    if head == _OLE_COMPOUND_MAGIC:
        try:
            return pd.read_excel(p, engine="xlrd", header=header)
        except ImportError as e:
            raise ImportError(
                "当前文件为 Excel 97-2003 二进制（.xls），需安装 xlrd：pip install xlrd"
            ) from e
    if head[:2] == b"PK":
        return pd.read_excel(p, engine="openpyxl", header=header)
    return read_csv_with_encoding_fallback(p, header=header, low_memory=low_memory)


def _resolve_metric_key(name_or_alias: str) -> str:
    if name_or_alias in METRIC_FILES:
        return name_or_alias
    for canon, aliases in METRIC_ALIAS.items():
        if name_or_alias == canon or name_or_alias in aliases:
            return canon
    raise KeyError(
        f"未注册指标 {name_or_alias!r}，请在 project_config.METRIC_FILES / METRIC_ALIAS 中配置。"
    )


def resolve_metric_csv_path(metric_name: str) -> Path:
    """将指标规范名或别名解析为宽表 CSV 的实际路径。"""
    key = _resolve_metric_key(metric_name)
    return WIDE_CSV_DIR / METRIC_FILES[key]


def metric_source_files_str(*logical_names: str) -> str:
    """将逻辑指标名解析为 `METRIC_FILES` 中的英文 CSV 文件名，去重后以逗号分隔（用于日志）。"""
    seen: set[str] = set()
    out: list[str] = []
    for raw in logical_names:
        key = _resolve_metric_key(raw)
        fn = METRIC_FILES[key]
        if fn not in seen:
            seen.add(fn)
            out.append(fn)
    return ", ".join(out)


def resolve_metric_asof_mode(metric_name: str) -> Literal["disclose", "date"] | None:
    """若指标为低频主序列，返回贴交易日时的对齐方式；否则 `None`。"""
    key = _resolve_metric_key(metric_name)
    return METRIC_ASOF_MODE.get(key)


def is_low_freq_metric(metric_name: str) -> bool:
    """是否为 `METRIC_ASOF_MODE` 中登记的低频主序列（规范名在 `METRIC_LOW_FREQ`）。"""
    key = _resolve_metric_key(metric_name)
    return key in METRIC_LOW_FREQ


def _strip_thousands_for_numeric(s: pd.Series) -> pd.Series:
    """宽表 melt 后常见「1,234.56」千分位；纯 `to_numeric` 会得到 NaN。统一去逗号并清洗占位符。"""
    t = s.astype(str).str.strip()
    t = t.mask(t.str.upper().isin(["#N/A", "N/A", "NA", "--", "NAN", "NONE", ""]), np.nan)
    t = t.str.replace(",", "", regex=False)
    return t


def _read_wide_csv(
    path: Path,
    *,
    period: Literal["date", "report_date"],
    date_col: Optional[str] = None,
    numeric_value: bool = True,
    report_value_as_text: bool = False,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"未找到 CSV: {path}")

    df = read_wide_metric_file(path, header=0, low_memory=False)
    time_name = "date" if period == "date" else "report_date"
    empty_cols = [time_name, "code", "value"]

    if df.empty or len(df.columns) < 2:
        return pd.DataFrame(columns=empty_cols)

    first_col = df.columns[0]
    tcol = date_col if date_col is not None and date_col in df.columns else first_col
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol])
    df = df.rename(columns={tcol: time_name})

    value_cols = [c for c in df.columns if c != time_name]
    if not value_cols:
        return pd.DataFrame(columns=empty_cols)

    long_df = pd.melt(
        df,
        id_vars=[time_name],
        value_vars=value_cols,
        var_name="code",
        value_name="value",
    )
    long_df["code"] = long_df["code"].astype(str).str.strip()

    if period == "report_date":
        if report_value_as_text:
            _bad = {"", "#N/A", "N/A", "NA", "--", "NAN", "NONE"}

            def _clean_disclose_text(x: object) -> object:
                if pd.isna(x):
                    return np.nan
                s = str(x).strip()
                if s.upper() in _bad or s in _bad:
                    return np.nan
                return s

            long_df["value"] = long_df["value"].map(_clean_disclose_text)
        else:
            long_df["value"] = pd.to_numeric(
                _strip_thousands_for_numeric(long_df["value"]), errors="coerce"
            )
            long_df["value"] = long_df["value"].replace([np.inf, -np.inf], np.nan)
        return long_df

    if numeric_value:
        long_df["value"] = pd.to_numeric(
            _strip_thousands_for_numeric(long_df["value"]), errors="coerce"
        )
        long_df["value"] = long_df["value"].replace([np.inf, -np.inf], np.nan)
    else:
        _bad = {"", "#N/A", "NA", "N/A", "--", "NAN", "NONE"}

        def _clean_cat(x):
            if pd.isna(x):
                return np.nan
            s = str(x).strip()
            if s.upper() in _bad or s in _bad:
                return np.nan
            return s

        long_df["value"] = long_df["value"].map(_clean_cat)
    return long_df


def read_unregistered_wide_csv(
    path: Path | str,
    *,
    period: Literal["date", "report_date"] = "report_date",
    **kwargs,
) -> pd.DataFrame:
    """未在 ``METRIC_FILES`` 登记的宽表文件路径，按与 :func:`read_metric` 相同规则 melt 为长表。"""
    return _read_wide_csv(Path(path), period=period, **kwargs)


def read_metric(metric_name: str, **kwargs) -> pd.DataFrame:
    """读取 `METRIC_FILES` 中注册的任意导出指标（日度/季度共用此接口）。"""
    key = _resolve_metric_key(metric_name)
    path = WIDE_CSV_DIR / METRIC_FILES[key]
    period: Literal["date", "report_date"] = (
        "report_date" if key in METRIC_USE_REPORT_DATE else "date"
    )
    extra: dict = {}
    if key in METRIC_WIDE_REPORT_VALUE_AS_TEXT:
        extra["report_value_as_text"] = True
    if key in METRIC_WIDE_DAILY_CATEGORICAL:
        extra["numeric_value"] = False
    return _read_wide_csv(path, period=period, **extra, **kwargs)


def apply_daily_missing_rules(df: pd.DataFrame, indicator: str) -> pd.DataFrame:
    """按指标类别应用既有日频缺失值与正值筛选规则。"""
    df = df.copy()
    delete_if_missing = {"收盘价", "总市值", "成交量", "收盘价(元)", "总市值(元)", "成交量(股)"}
    ffill_if_missing = {"流通股本", "A股流通股本(股)"}

    if indicator in delete_if_missing:
        df = df.dropna(subset=["value"])
        df = df[df["value"] > 0]
    elif indicator in ffill_if_missing:
        df = df.sort_values(["code", "date"])
        df["value"] = df.groupby("code")["value"].ffill()
        df = df.dropna(subset=["value"])
        df = df[df["value"] > 0]
    else:
        df = df.dropna(subset=["value"])
    return df.reset_index(drop=True)


def round_factor_output(df: pd.DataFrame, decimals: int = 6) -> pd.DataFrame:
    """仅舍入数值列，保持非数值列和输入列顺序不变。"""
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        df = df.copy()
        df[num_cols] = df[num_cols].round(decimals)
    return df


def get_daily_date_code_grid() -> pd.DataFrame:
    """从收盘价数据取得去重后的交易日与标的基础网格。"""
    df = read_metric("收盘价")
    return df[["date", "code"]].drop_duplicates()


def build_disclose_date(
    q_df: pd.DataFrame,
    actual: Optional[pd.DataFrame] = None,
    expected: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """按实际、预计、报告期后 60 日的优先级确定披露可得日。"""
    out = q_df[["report_date", "code"]].drop_duplicates().copy()
    default_dt = pd.to_datetime(out["report_date"]) + pd.Timedelta(days=60)

    if actual is not None and len(actual) > 0:
        ad = actual.copy()
        ad["actual_dt"] = pd.to_datetime(ad["value"], errors="coerce")
        out = out.merge(ad[["report_date", "code", "actual_dt"]], on=["report_date", "code"], how="left")
    else:
        out["actual_dt"] = pd.NaT

    if expected is not None and len(expected) > 0:
        ed = expected.copy()
        ed["exp_dt"] = pd.to_datetime(ed["value"], errors="coerce")
        out = out.merge(ed[["report_date", "code", "exp_dt"]], on=["report_date", "code"], how="left")
    else:
        out["exp_dt"] = pd.NaT

    out["disclose_date"] = out["actual_dt"].fillna(out["exp_dt"]).fillna(default_dt)
    return out[["report_date", "code", "disclose_date"]]


def lowfreq_to_daily_asof(
    df_low: pd.DataFrame,
    time_col: str,
    value_col: str,
    daily_dates: pd.DataFrame,
) -> pd.DataFrame:
    """将低频长表贴到交易日网格：对每个 (date, code) 取截至该日最近一条观测（`merge_asof` backward）。

    **季频、月频、年频共用此实现**，不限于季报。`time_col` 可为观测日 `date`（如月末）、
    `disclose_date`（披露可得日）等，由调用方与 `METRIC_ASOF_MODE` 约定一致即可。

    贴日前会丢弃 `value_col` 为 NaN 的行，使其不参与 as-of；避免「仅有披露日、无数值」的占位行
    在 backward 下成为最近一条观测并覆盖此前有效值。
    """
    if "code" in daily_dates.columns:
        daily = daily_dates[["date", "code"]].drop_duplicates()
    else:
        dates = daily_dates["date"].unique()
        codes = df_low["code"].unique()
        daily = pd.DataFrame([(d, c) for d in dates for c in codes], columns=["date", "code"])
    daily["date"] = pd.to_datetime(daily["date"]).astype("datetime64[ns]")
    df_right = df_low[["code", time_col, value_col]].copy()
    df_right[time_col] = pd.to_datetime(df_right[time_col]).astype("datetime64[ns]")
    # 仅把「该指标有有效值」的观测放进 as-of 时间轴。否则（如 Profitability 中外连合并后
    # 季频行上 GP 为 NaN 仍有 disclose_date）backward 会选到更晚的「空披露」，覆盖较早的有效年报。
    df_right = df_right.dropna(subset=[value_col])
    df_right = df_right.drop_duplicates(subset=["code", time_col], keep="last")
    daily = daily.sort_values("date").reset_index(drop=True)
    df_right = df_right.sort_values(time_col).reset_index(drop=True)
    merged = pd.merge_asof(
        daily,
        df_right,
        left_on="date",
        right_on=time_col,
        by="code",
        direction="backward",
    )
    out = merged.dropna(subset=[value_col])[["date", "code", value_col]]
    return out.sort_values(["date", "code"]).reset_index(drop=True)


def _find_column(codes: list, name_or_alias: str) -> Optional[str]:
    codes = [str(c).strip() for c in codes]
    if name_or_alias in codes:
        return name_or_alias
    for canonical, aliases in METRIC_ALIAS.items():
        if name_or_alias == canonical or name_or_alias in aliases:
            for c in codes:
                if c == canonical or c in aliases or canonical in c:
                    return c
    for c in codes:
        if name_or_alias in c or c in name_or_alias:
            return c
    return None


def get_index_returns(index_column: Optional[str] = None) -> pd.Series:
    """读取指定基准指数点位并计算按日期索引的日收益率。"""
    df = read_metric(INDEX_SHEET)
    if df.empty:
        p = resolve_metric_csv_path(INDEX_SHEET)
        raise ValueError(
            f"指标「{INDEX_SHEET}」无有效数据（{p} 为空或日期列无法解析）。"
            f"请检查文件编码（建议 UTF-8 / GBK）且首列为日期、其余列为指数点位。"
        )

    use_column = index_column if index_column is not None else INDEX_COLUMN
    actual_col = _find_column(df["code"].unique().tolist(), use_column)
    if actual_col is None:
        raise KeyError(
            f"基准指数 CSV 中未找到列「{use_column}」。可用 code: {df['code'].unique().tolist()}"
        )
    df = df[df["code"] == actual_col]
    out = df[["date", "value"]].copy()
    out = out.sort_values("date").dropna(subset=["value"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"])
    out["index_ret"] = out["value"].pct_change()
    out = out.dropna(subset=["index_ret"])
    return out.set_index("date")["index_ret"]


class DataIO:
    """统一原始宽表、分析输入和基准数据读取。"""

    def __init__(self, data_dir: Path | str | None = None, config=default_config) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else Path(config.WIDE_CSV_DIR)
        self.config = config

    def _metric_key(self, name_or_alias: str) -> str:
        files = self.config.METRIC_FILES
        if name_or_alias in files:
            return name_or_alias
        for canonical, aliases in self.config.METRIC_ALIAS.items():
            if name_or_alias == canonical or name_or_alias in aliases:
                return canonical
        raise KeyError(f"未注册指标 {name_or_alias!r}，请在 project_config.METRIC_FILES / METRIC_ALIAS 中配置。")

    def resolve_metric_path(self, metric_name: str) -> Path:
        key = self._metric_key(metric_name)
        return self.data_dir / self.config.METRIC_FILES[key]

    def metric_source_files_str(self, *logical_names: str) -> str:
        seen: set[str] = set()
        names: list[str] = []
        for raw in logical_names:
            filename = self.config.METRIC_FILES[self._metric_key(raw)]
            if filename not in seen:
                seen.add(filename)
                names.append(filename)
        return ", ".join(names)

    def read_metric(self, metric_name: str, **kwargs) -> pd.DataFrame:
        key = self._metric_key(metric_name)
        period: Literal["date", "report_date"] = (
            "report_date" if key in self.config.METRIC_USE_REPORT_DATE else "date"
        )
        extra: dict = {}
        if key in self.config.METRIC_WIDE_REPORT_VALUE_AS_TEXT:
            extra["report_value_as_text"] = True
        if key in self.config.METRIC_WIDE_DAILY_CATEGORICAL:
            extra["numeric_value"] = False
        return _read_wide_csv(self.resolve_metric_path(key), period=period, **extra, **kwargs)

    def apply_missing_rules(self, df: pd.DataFrame, indicator: str) -> pd.DataFrame:
        return apply_daily_missing_rules(df, indicator)

    def get_daily_date_code_grid(self) -> pd.DataFrame:
        df = self.read_metric("收盘价")
        return df[["date", "code"]].drop_duplicates()

    def read_optional_metric(self, metric_name: str) -> Optional[pd.DataFrame]:
        try:
            return self.read_metric(metric_name)
        except (KeyError, FileNotFoundError, OSError):
            return None

    def read_disclose_pair(self) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        return self.read_optional_metric("实际披露日期"), self.read_optional_metric("预计披露日期")

    def attach_disclose_date(
        self,
        q_df: pd.DataFrame,
        actual: Optional[pd.DataFrame],
        expected: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        disc = self.build_disclose_date(q_df, actual, expected)
        out = q_df.merge(disc, on=["report_date", "code"], how="left")
        out["disclose_date"] = pd.to_datetime(out["disclose_date"], errors="coerce")
        return out.dropna(subset=["disclose_date"])

    def quarterly_values_to_daily(
        self,
        q_df: pd.DataFrame,
        value_cols: list[str],
        daily_grid: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if daily_grid is None:
            daily_grid = self.get_daily_date_code_grid()
        actual, expected = self.read_disclose_pair()
        df = self.attach_disclose_date(q_df, actual, expected)
        for column in value_cols:
            if column not in df.columns:
                raise KeyError(f"季度表缺少列 {column!r}")
        merged: Optional[pd.DataFrame] = None
        for value_col in value_cols:
            piece = self.lowfreq_to_daily(df, "disclose_date", value_col, daily_grid)
            merged = piece if merged is None else merged.merge(piece, on=["date", "code"], how="outer")
        assert merged is not None
        return merged.sort_values(["date", "code"]).reset_index(drop=True)

    def _merge_metric(
        self,
        daily: pd.DataFrame,
        metric_name: str,
        rename_to: str,
        *,
        positive: bool = False,
    ) -> pd.DataFrame:
        metric = self.apply_missing_rules(self.read_metric(metric_name), metric_name)
        metric = metric.rename(columns={"value": rename_to})
        out = daily.merge(metric, on=["date", "code"], how="inner")
        return out[out[rename_to] > 0] if positive else out

    def merge_mkt_cap(self, daily: pd.DataFrame) -> pd.DataFrame:
        return self._merge_metric(daily, "总市值", "mkt_cap", positive=True)

    def merge_close(self, daily: pd.DataFrame) -> pd.DataFrame:
        return self._merge_metric(daily, "收盘价", "price")

    @staticmethod
    def lag_mkt_cap_one_trading_day(df: pd.DataFrame) -> pd.DataFrame:
        out = df.sort_values(["code", "date"]).copy()
        out["mkt_cap"] = out.groupby("code", sort=False)["mkt_cap"].shift(1)
        return out.dropna(subset=["mkt_cap"]).loc[lambda x: x["mkt_cap"] > 0]

    def nan_descriptors_from_daily(self, descriptor_cols: list[str]) -> pd.DataFrame:
        out = self.get_daily_date_code_grid().copy()
        for column in descriptor_cols:
            out[column] = np.nan
        return out.sort_values(["date", "code"]).reset_index(drop=True)

    def load_factors(self, path: Path | None = None) -> pd.DataFrame:
        p = path or self.config.FACTORS_MERGED
        if not Path(p).exists():
            raise FileNotFoundError(f"未找到因子文件: {p}")
        df = pd.read_csv(p, encoding="utf-8-sig")
        rename = {k: v for k, v in self.config.MERGED_COLUMN_TO_FACTOR_CODE.items() if k in df.columns}
        if rename:
            df = df.rename(columns=rename)
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= self.config.ANALYSIS_START) & (df["date"] <= self.config.ANALYSIS_END)]
        return df.sort_values(["date", "code"]).reset_index(drop=True)

    def load_returns(self, path: Path | None = None) -> pd.DataFrame:
        df = self.apply_missing_rules(self.read_metric("收盘价"), "收盘价")
        df = df.rename(columns={"value": "close"}).sort_values(["code", "date"])
        df["ret"] = df.groupby("code")["close"].pct_change()
        df = df.dropna(subset=["ret"]).drop(columns=["close"])
        df = df[(df["date"] >= self.config.ANALYSIS_START) & (df["date"] <= self.config.ANALYSIS_END)]
        return df[["date", "code", "ret"]].reset_index(drop=True)

    def load_industry(self, path: Path | None = None) -> pd.DataFrame:
        try:
            df = self.read_metric("行业")
        except (FileNotFoundError, KeyError, OSError):
            return pd.DataFrame(columns=["date", "code", "industry"])
        df = df.rename(columns={"value": "industry"})
        df["date"] = pd.to_datetime(df["date"])
        df["code"] = df["code"].astype(str).str.strip()
        df = df[(df["date"] >= self.config.ANALYSIS_START) & (df["date"] <= self.config.ANALYSIS_END)]
        return df[["date", "code", "industry"]].reset_index(drop=True)

    def load_market_cap(self, path: Path | None = None) -> pd.DataFrame:
        df = self.apply_missing_rules(self.read_metric("总市值"), "总市值")
        df = df.rename(columns={"value": "mkt_cap"})
        df = df[(df["date"] >= self.config.ANALYSIS_START) & (df["date"] <= self.config.ANALYSIS_END)]
        return df[["date", "code", "mkt_cap"]].reset_index(drop=True)

    def load_benchmark_returns(self, index_column: Optional[str] = None) -> pd.Series:
        df = self.read_metric(self.config.INDEX_SHEET)
        if df.empty:
            raise ValueError(f"指标「{self.config.INDEX_SHEET}」无有效数据（{self.resolve_metric_path(self.config.INDEX_SHEET)}）。")
        use_column = index_column if index_column is not None else self.config.INDEX_COLUMN
        actual_col = _find_column(df["code"].unique().tolist(), use_column)
        if actual_col is None:
            raise KeyError(f"基准指数 CSV 中未找到列「{use_column}」。可用 code: {df['code'].unique().tolist()}")
        out = df[df["code"] == actual_col][["date", "value"]].copy()
        out = out.sort_values("date").dropna(subset=["value"])
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out.dropna(subset=["value"])
        out["index_ret"] = out["value"].pct_change()
        return out.dropna(subset=["index_ret"]).set_index("date")["index_ret"]

    @staticmethod
    def build_disclose_date(
        q_df: pd.DataFrame,
        actual: Optional[pd.DataFrame] = None,
        expected: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        return build_disclose_date(q_df, actual, expected)

    @staticmethod
    def lowfreq_to_daily(
        df_low: pd.DataFrame,
        time_col: str,
        value_col: str,
        daily_dates: pd.DataFrame,
    ) -> pd.DataFrame:
        return lowfreq_to_daily_asof(df_low, time_col, value_col, daily_dates)

    @staticmethod
    def attach_prior_month_end_close(
        daily: pd.DataFrame,
        price: pd.DataFrame,
        *,
        date_col: str = "date",
        code_col: str = "code",
        price_col: str = "price",
        out_col: str = "price_prior_month_end",
    ) -> pd.DataFrame:
        p = price[[date_col, code_col, price_col]].copy()
        p[date_col] = pd.to_datetime(p[date_col])
        p = p.sort_values([code_col, date_col])
        p["y"], p["m"] = p[date_col].dt.year, p[date_col].dt.month
        month_end = p.groupby([code_col, "y", "m"], as_index=False).last()
        month_end = month_end[[code_col, "y", "m", price_col]].rename(
            columns={"y": "_py", "m": "_pm", price_col: out_col}
        )
        out = daily.copy()
        out[date_col] = pd.to_datetime(out[date_col])
        out["_py"] = np.where(out[date_col].dt.month == 1, out[date_col].dt.year - 1, out[date_col].dt.year)
        out["_pm"] = np.where(out[date_col].dt.month == 1, 12, out[date_col].dt.month - 1)
        return out.merge(month_end, on=[code_col, "_py", "_pm"], how="left").drop(columns=["_py", "_pm"])


__all__ = [
    "INDEX_COLUMN",
    "INDEX_SHEET",
    "WIDE_CSV_DIR",
    "apply_daily_missing_rules",
    "read_csv_with_encoding_fallback",
    "read_wide_metric_file",
    "build_disclose_date",
    "get_daily_date_code_grid",
    "get_index_returns",
    "is_low_freq_metric",
    "lowfreq_to_daily_asof",
    "metric_source_files_str",
    "read_metric",
    "resolve_metric_asof_mode",
    "resolve_metric_csv_path",
    "round_factor_output",
    "DataIO",
]
