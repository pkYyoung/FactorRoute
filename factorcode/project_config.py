"""
项目级共用配置（单一事实来源）。

包含：因子列清单、路径、样本分析期、持仓周期、**时间切片导出粒度**（`TEMPORAL_EXPORT_FREQS`）、IC 时间切片周期锚点（`TEMPORAL_IC_LABEL_ANCHOR`）等。
日频原始数据为 **每指标一 CSV 宽表**（见 ``data.io`` 与《原始数据标准规范》）。
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent

# ========= 因子列（回测 / IC / 绘图等默认跑哪些因子；增删因子只改此处）=========
FACTOR_COLUMNS = [
    "SIZE", "MOM", "BETA", "LIQUID", "LREV", "RESVOL",
    "MIDCAP", "PROFIT", "LEVERAGE", "EQLTY", "INVQ",
    "GROWTH", "EVAR", "BTOP", "EYLD", "DIVYLD",
]

# 合并表 / 相关矩阵的「磁盘列名」→ 全项目统一简称（``FACTOR_COLUMNS`` 子集）
#
# 设计要点：
# - 回测/IC/作图只认 ``FACTOR_COLUMNS`` 等字符串；计算侧``style_factors_merged`` 可能是一级
#   ``style_key``（如 Size）或仅二级 ``descriptor``（如 LNCAP、STOM），本表在**读入时**做重命名对齐。
# - 键应能在 ``data.factors`` 的因子元数据中查到：一级用 ``style_key``，二级用 ``descriptors`` 中的名字。
# - 同一张表内**勿**让两列映射到**同一**目标简称（会列名冲突）；仅跑二级、且用「单 descriptor
#   代理某一风格」时，在此增加一行，例如 ``"LNCAP": "SIZE"``，并视情况收紧 ``FACTOR_COLUMNS``。
MERGED_COLUMN_TO_FACTOR_CODE: dict[str, str] = {
    "Size": "SIZE",
    "Momentum": "MOM",
    "Beta": "BETA",
    "Liquidity": "LIQUID",
    "LongTermReversal": "LREV",
    "ResidualVolatility": "RESVOL",
    "MidCapitalization": "MIDCAP",
    "Profitability": "PROFIT",
    "Leverage": "LEVERAGE",
    "EarningsQuality": "EQLTY",
    "InvestmentQuality": "INVQ",
    "Growth": "GROWTH",
    "EarningsVariability": "EVAR",
    "BookToPrice": "BTOP",
    "EarningsYield": "EYLD",
    "DividendYield": "DIVYLD",
    # 二级列名在此按需追加，例如只合并 descriptor 时： "LNCAP": "SIZE",
}

# ========= 路径 =========
WORKSPACE_ROOT = Path(os.getenv("FACTORROUTE_HOME", str(PROJECT_ROOT.parent))).expanduser().resolve()
DATA_RAW = WORKSPACE_ROOT / "data" / "raw"
DATA_OUT = WORKSPACE_ROOT / "outputs"
FACTORS_MERGED = DATA_OUT / "style_factors_merged.csv"
FACTOR_CORRELATION_CSV = DATA_OUT / "style_factor_correlation.csv"
WIDE_CSV_DIR = Path(os.getenv("FC_WIDE_CSV_DIR", str(DATA_RAW)))

# ========= 样本期：默认不过滤常见日期，正式实验必须显式设置 FC_ANALYSIS_START/END =========
ANALYSIS_START = os.getenv("FC_ANALYSIS_START", "1900-01-01")
ANALYSIS_END = os.getenv("FC_ANALYSIS_END", "2100-12-31")

# ========= 持仓 / 周期（与 IC、排序回测一致）=========
# 'W'=周度，'M'=月度
HOLDING_PERIOD = "W"
PERIOD_FREQ = "W-FRI" if HOLDING_PERIOD == "W" else "M"

# ========= 时间切片导出（run_ic / run_backtest 末尾共用 temporal_export）=========
# 可选: "week" / "month" / "quarter"；空元组 () 表示不自动导出。
TEMPORAL_EXPORT_FREQS: tuple[str, ...] = ("month",)

# ========= IC 时间切片周期标签（run_ic / temporal_export 共用）=========
# 仅影响 IC 切片周期标签；多空切片不受此锚点影响。
TEMPORAL_IC_LABEL_ANCHOR: str = "start"
# 读数/登记约定（后续同类指标请沿用）：
# - 单期、TTM、年报等多套口径：在 METRIC_FILES 中用「指标_单期」「指标_TTM」「指标_年报」等完整逻辑名分别注册；
#   代码里 read_metric / read_optional_metric 只写完整名，不写简称。
# - 英文文件名建议 *_single_quarterly / *_ttm_quarterly / *_annual_quarterly，与逻辑名一一对应。
# - 「实际披露日期」「预计披露日期」米筐无数据时由 Wind 等导出，仍放在下表对应 CSV。

# 导出指标逻辑名 → 文件名（相对 WIDE_CSV_DIR）；读入时不区分「频率」，只认宽表形态
METRIC_FILES: dict[str, str] = {
    "收盘价": "close_daily.csv",
    # 日度 OHLC 与换手率（宽表，供行情与后续执行研究使用）
    "开盘价": "open_daily.csv",
    "最高价": "high_daily.csv",
    "最低价": "low_daily.csv",
    "总市值": "mktcap_total_daily.csv",
    "成交量": "volume_daily.csv",
    "流通股本": "float_shares_daily.csv",
    # 换手率（Wind s_dq_turn：通常为“百分数(%)”口径；不在代码内二次缩放）
    "换手率": "turnover_rate_daily.csv",
    "流通股本_财报": "float_shares_quarterly.csv",
    "流通股本_年报": "float_shares_annual_quarterly.csv",
    "基准指数": "benchmark_index_daily.csv",
    # 日度行业分类（宽表；单元格为行业名/代码字符串，走 ``read_metric`` 时按类别列读取）
    "行业": "industry_daily.csv",
    # DTOP：宽表 value = 过去12个月每股现金股利（元/股）；输入须为每股金额而非股息率
    "过去12个月现金股利": "dividend_ttm_quarterly.csv",
    "实际披露日期": "disclose_actual_quarterly.csv",
    "预计披露日期": "disclose_expected_quarterly.csv",
    # DTOPF：Wind s_west_avgdps_fy1 为 FY1 一致预期每股股利；12m 与附录「预测 12 个月股息」对应，季末宽表见 Barra 整理稿 §2
    "预测每股股利": "fc_dps_12m_quarterly.csv",
    # 财报类（季度宽表；与 Wind/米筐导出字段一一对应，文件名可自定，须在此登记）
    "股东权益": "equity_single_quarterly.csv",
    "股东权益_年报": "equity_annual_quarterly.csv",
    "净利润_TTM": "ni_ttm_quarterly.csv",
    "净利润_单期": "ni_single_quarterly.csv",
    "净利润_年报": "ni_annual_quarterly.csv",
    "总资产": "total_assets_single_quarterly.csv",
    "总资产_年报": "total_assets_annual_quarterly.csv",
    "总负债": "total_liab_single_quarterly.csv",
    "总负债_年报": "total_liab_annual_quarterly.csv",
    "营业收入_TTM": "revenue_ttm_quarterly.csv",
    "营业收入_年报": "revenue_annual_quarterly.csv",
    "营业成本_单期": "cogs_single_quarterly.csv",
    "营业成本_TTM": "cogs_ttm_quarterly.csv",
    "营业成本_年报": "cogs_annual_quarterly.csv",
    # Growth（compute_growth）：与 EGRO/SGRO 实现一致，用 LYR 年报季末宽表（与 basic_eps_lyr / operating_revenue_per_share_lyr 同文件）
    "每股收益": "basic_eps_lyr_annual_quarterly.csv",
    "每股营业收入": "operating_revenue_per_share_lyr_annual_quarterly.csv",
    "基本每股收益_年报": "basic_eps_lyr_annual_quarterly.csv",
    "每股营业收入_年报": "operating_revenue_per_share_lyr_annual_quarterly.csv",
    "营业收入_单期": "revenue_single_quarterly.csv",
    "经营活动现金流": "cfo_single_quarterly.csv",
    "经营活动现金流_年报": "cfo_annual_quarterly.csv",
    "投资活动现金流": "cfi_single_quarterly.csv",
    "投资活动现金流_年报": "cfi_annual_quarterly.csv",
    "折旧摊销": "da_single_quarterly.csv",
    "折旧摊销_年报": "da_annual_quarterly.csv",
    "现金及现金等价物": "cash_single_quarterly.csv",
    # VFLO：附录为年「现金及现金等价物净增加额」；导出须与数据源字段一致（米筐 cash_equivalent_increase 等）
    "现金及现金等价物_年报": "cash_annual_quarterly.csv",
    "优先股": "preferred_stock_single_quarterly.csv",
    "优先股_年报": "preferred_stock_annual_quarterly.csv",
    "长期负债": "long_debt_single_quarterly.csv",
    "长期负债_年报": "long_debt_annual_quarterly.csv",
    "带息债务": "interest_bearing_debt_single_quarterly.csv",
    "带息债务_年报": "interest_bearing_debt_annual_quarterly.csv",
    # CETOP 分子：Barra「现金盈利」实务代理为经营现金流 TTM；Wind 可用 s_fa_operatecashflow_ttm，按季末导出宽表
    "现金盈利_TTM": "cash_earnings_ttm_quarterly.csv",
    # EM：Barra「上一财政年度 EBIT」；Wind 按年报报告期导出 → 与 ni_annual_quarterly 同系命名
    "息税前利润_年报": "ebit_annual_quarterly.csv",
    "企业价值": "enterprise_value_daily.csv",
    "资本支出": "capex_quarterly.csv",
    # CXGRO：Wind 现金流量表「购建固定资产、无形资产和其他长期资产支付的现金」年报宽表；勿用 capex/DA 比率代替金额序列
    "资本支出_年报": "capex_annual_quarterly.csv",
    "分析师预测EPS": "fc_eps_12m_quarterly.csv",
    # EGRLF（compute_growth）：与「净利润_年报」合并算 (FY3/NI)^(1/3)-1；本键与下行同文件，供旧逻辑名或外部脚本读取 FY3 宽表
    "分析师预测长期盈利增长": "fc_net_profit_fy3_annual_quarterly.csv",
    "分析师预测盈利": "fc_profit_12m_quarterly.csv",
    # Wind s_est_stdeps 等为「对某一财年 EPS 预测分歧」；宽表形态与 ni_annual_quarterly 同系，命名用 *_annual_quarterly
    "分析师预测EP标准差": "fc_eps_std_annual_quarterly.csv",
    # Wind 一致预期等：FY3 归母净利润预测（年频宽表；命名沿用 *_annual_quarterly 与 ni_annual_quarterly 一致）
    "一致预期净利润_FY3": "fc_net_profit_fy3_annual_quarterly.csv",
}

# 规范名 → 同义名（解析到 METRIC_FILES 键）
METRIC_ALIAS: dict[str, tuple[str, ...]] = {
    "收盘价": ("收盘价", "收盘价(元)", "复权收盘价"),
    "开盘价": ("开盘价", "开盘价(元)", "复权开盘价"),
    "最高价": ("最高价", "最高价(元)", "复权最高价"),
    "最低价": ("最低价", "最低价(元)", "复权最低价"),
    "总市值": ("总市值", "总市值(元)"),
    "成交量": ("成交量", "成交量(股)"),
    "流通股本": ("流通股本", "A股流通股本(股)"),
    "换手率": ("换手率", "换手率(%)", "日换手率"),
    "基准指数": ("基准指数",),
    "行业": ("行业", "申万行业", "申万一级"),
    "过去12个月现金股利": ("过去12个月现金股利", "现金股利", "派息"),
    "息税前利润_年报": ("息税前利润",),
}

# 下列指标 melt 后时间列名为 report_date（财报/季度语义）；其余为 date（交易日语义）
_METRIC_DATE_SEMANTIC: frozenset[str] = frozenset(METRIC_FILES.keys()) - frozenset(
    {
        "收盘价",
        "开盘价",
        "最高价",
        "最低价",
        "总市值",
        "成交量",
        "流通股本",
        "换手率",
        "基准指数",
        "企业价值",
        "行业",
    }
)
METRIC_USE_REPORT_DATE: frozenset[str] = _METRIC_DATE_SEMANTIC

# 宽表 melt 后 `value` 为日期字符串（如 1998-08-10）或 #N/A，**不能**走 `pd.to_numeric`（否则全变 NaN）
METRIC_WIDE_REPORT_VALUE_AS_TEXT: frozenset[str] = frozenset({"实际披露日期", "预计披露日期"})

# 日频宽表但值为类别/字符串（如行业名），禁止当数值解析
METRIC_WIDE_DAILY_CATEGORICAL: frozenset[str] = frozenset({"行业"})

# 低频「主序列」：相对交易日更稀，需贴到交易日网格（merge_asof 等）。
# - "disclose"：先有 report_date，再经 build_disclose_date 得 disclose_date 后对齐（防前视）。
# - "date"：长表已有交易日语义列 date（如月末观测），直接作为 as-of 时间轴对齐。
# 「实际披露日期」「预计披露日期」等为披露辅助表，不设此项；见各 compute 脚本。
_METRIC_ASOF_DISCLOSE: frozenset[str] = frozenset(
    k
    for k in METRIC_FILES
    if k
    not in {
        "收盘价",
        "开盘价",
        "最高价",
        "最低价",
        "总市值",
        "成交量",
        "流通股本",
        "换手率",
        "基准指数",
        "企业价值",
        "行业",
        "实际披露日期",
        "预计披露日期",
    }
)
METRIC_ASOF_MODE: dict[str, Literal["disclose", "date"]] = {
    **{k: "disclose" for k in _METRIC_ASOF_DISCLOSE},
    "企业价值": "date",
}
METRIC_LOW_FREQ: frozenset[str] = frozenset(
    k for k, v in METRIC_ASOF_MODE.items() if v == "disclose"
)

for _m in METRIC_ASOF_MODE:
    if _m not in METRIC_FILES:
        raise ValueError(f"METRIC_ASOF_MODE 键 {_m!r} 须在 METRIC_FILES 中注册")

# 基准指数 CSV（逻辑名「基准指数」）中作为列名的指数代码
INDEX_SHEET = "基准指数"
INDEX_COLUMN = os.getenv("FC_BENCHMARK", "BENCHMARK")

# Report：基准在图例中的「简称(列名)」
INDEX_BENCHMARK_DISPLAY_SHORT: dict[str, str] = {}


def index_benchmark_chart_label(column_name: str | None) -> str:
    """
    将基准列名格式化为「指数简称(列名)」，用于绘图；未命中映射时返回原列名。
    已含括号（如手工传入的展示串）则原样返回，避免重复包裹。
    """
    if column_name is None or not str(column_name).strip():
        return "基准"
    c = str(column_name).strip()
    if "(" in c and ")" in c:
        return c
    short = INDEX_BENCHMARK_DISPLAY_SHORT.get(c)
    if short is None:
        return c
    if short == c:
        return c
    return f"{short}({c})"


BENCHMARK_ORDER_BOOK_ID = os.getenv("FC_BENCHMARK", "BENCHMARK")
UNIVERSE_MODE = os.getenv("FC_UNIVERSE", "index")  # "index" | "all"

DATA_FACTORS = DATA_OUT / "factors"
FACTOR_OUTPUT_DECIMALS = 6
FACTOR_DECIMALS = int(os.getenv("FC_FACTOR_DECIMALS", str(FACTOR_OUTPUT_DECIMALS)))

# ---------------------------------------------------------------------------
# 因子计算常用窗口（半衰、滚动长度等，可按模型调整）
# ---------------------------------------------------------------------------
HBETA_WINDOW = 252
HBETA_HALFLIFE = 63
ATVR_WINDOW = 252
ATVR_HALFLIFE = 63
# DASTD：日收益半衰加权波动（渤海/Barra 常见 252/42）
DASTD_WINDOW = 252
DASTD_HALFLIFE = 42
# 流动性 STOM/STOQ/STOA：与 STOM 同构，均为「过去 N 个交易日日换手率之和再取自然对数」。
STOM_DAYS = 21
STOQ_DAYS = 63   # 约一季交易日
STOA_DAYS = 252  # 约一年交易日
RSTR_WINDOW = 252
RSTR_HALFLIFE = 126
LTRSTR_WINDOW = 1040
LTRSTR_HALFLIFE = 260
LTRSTR_LAG = 273


# ========= 因子运行目标与一级合成参数 =========
RunTarget = str
MergeMethod = Literal["zscore_weighted", "weighted_raw"]

@dataclass
class Level1MergeConfig:
    """一级合成的权重与方法。"""

    weights: dict[str, float]
    method: MergeMethod = "zscore_weighted"

RUN_TARGETS: dict[str, str] = {
    name: name for name in (
        "Size", "Beta", "Liquidity", "Momentum", "ResidualVolatility",
        "MidCapitalization", "Profitability", "Leverage", "EarningsQuality",
        "InvestmentQuality", "Growth", "EarningsVariability", "BookToPrice",
        "EarningsYield", "DividendYield", "LongTermReversal",
    )
}

LEVEL1_WEIGHTED: dict[str, list[float]] = {
    "Size": [1.0], "Beta": [1.0], "Liquidity": [0.25] * 4,
    "Momentum": [0.5] * 2, "ResidualVolatility": [1.0 / 3.0] * 3,
    "MidCapitalization": [1.0], "Profitability": [0.25] * 4,
    "Leverage": [1.0 / 3.0] * 3, "EarningsQuality": [0.5] * 2,
    "InvestmentQuality": [1.0 / 3.0] * 3, "Growth": [1.0 / 3.0] * 3,
    "EarningsVariability": [0.25] * 4, "BookToPrice": [1.0],
    "EarningsYield": [0.25] * 4, "DividendYield": [0.5] * 2,
    "LongTermReversal": [0.5] * 2,
}
LEVEL1_METHOD: dict[str, MergeMethod] = {"MidCapitalization": "weighted_raw"}

def parse_run_segment(seg: str) -> tuple[str, RunTarget]:
    """解析 ``大类:目标``；目标是否属于描述子由对应因子类校验。"""
    value = seg.strip()
    if ":" not in value:
        raise ValueError(f"须为「大类:目标」形式，得到 {value!r}")
    style_key, target = (part.strip() for part in value.split(":", 1))
    if style_key not in RUN_TARGETS:
        raise ValueError(f"未知大类 {style_key!r}，允许: {sorted(RUN_TARGETS)}")
    if not target:
        raise ValueError(f"目标不能为空：{value!r}")
    return style_key, target

def _parse_env_run_plan(raw: str) -> list[tuple[str, str]]:
    return [parse_run_segment(seg) for seg in raw.split(";") if seg.strip()]

def parse_run_cli_tokens(tokens: Sequence[str]) -> list[tuple[str, str]]:
    return [parse_run_segment(token) for token in tokens if str(token).strip()]

def get_run_plan() -> list[tuple[str, str]]:
    raw = os.getenv("FC_RUN_STYLES", "").strip()
    return _parse_env_run_plan(raw) if raw else list(RUN_TARGETS.items())

def get_target_for_style(style_key: str) -> RunTarget:
    raw = os.getenv("FC_RUN_STYLES", "").strip()
    if raw:
        for key, target in _parse_env_run_plan(raw):
            if key == style_key:
                return target
    if style_key not in RUN_TARGETS:
        raise KeyError(f"{style_key!r} 不在 RUN_TARGETS 中")
    return RUN_TARGETS[style_key]

# ========= Analysis 与绘图参数 =========
DATA_IC = DATA_OUT / "ic"
DATA_BACKTEST = DATA_OUT / "backtest"
IC_CORR_TYPE = "spearman"
TOP_PCT = 0.2
BOTTOM_PCT = 0.2
WEIGHT_METHOD = "市值"
PREPROCESS_PIPELINE: tuple = ("winsor", "zscore")
WINSOR_MAD_SCALE: float = 3.0
INDUSTRY_SHEET_NAME = "行业"
SIZE_NEUTRAL_EXCLUDE: tuple = ("SIZE", "MIDCAP")
STACKED_BARS_FREQ: str = "month"
BACKTEST_BENCHMARK_INDEX_COLUMN: str = INDEX_COLUMN
DEFAULT_PHASES_JSON = PROJECT_ROOT / "configs" / "phases" / "phases_example.json"
