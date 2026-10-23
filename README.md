# FactorRoute

面向 2026 量化交易大赛的周频股票多因子研究项目。策略目标为无杠杆股票多头：周五收盘后生成信号，按固定多因子评分构建组合，在下一周首个交易日开盘阶段调仓。

当前实现候选因子计算、预处理、IC 与分组分析；赛事数据接入、评分组合及交易回测尚待开发。

## 功能

- 16 个 Barra 风格候选因子，覆盖规模、价值、成长、质量、动量、波动和流动性。
- 行情与财务数据读取、披露日对齐、MAD 缩尾、标准化及可选中性化。
- IC、因子相关性、Top/Bottom 分组收益、绩效统计与图表。

## 项目结构

```text
.
├── factorcode/                 # Python 研究工具包
│   ├── data/                  # 数据读取、因子公式与计算流程
│   ├── analysis/              # IC、分组研究、指标与图表
│   ├── configs/phases/        # 自定义分析区间模板
│   ├── project_config.py      # 统一研究配置
│   └── research_engine.py     # 命令行与流程入口
├── data/raw/                  # 本地原始输入
├── outputs/                   # 计算与分析结果
├── experiments/               # 实验配置、数据版本与结论
├── scripts/verify_research.py  # 合成数据链路检查
└── pyproject.toml             # 安装配置与依赖
```

根目录 `data/` 存放数据文件，`factorcode/data/` 存放处理代码。[模块与因子说明](factorcode/README.md)见包内 README。

## 快速开始

推荐 Python 3.12，支持 Python 3.10–3.12。在项目根目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/verify_research.py
```

Windows 使用 `py -3.12 -m venv .venv` 创建环境，通过 `.venv\Scripts\activate.bat`（CMD）或 `.venv\Scripts\Activate.ps1`（PowerShell）激活。

验证使用临时合成数据，覆盖核心导入、Size 公式及 IC/分组输出，无需外部账号。读取 Excel 输入时，可使用 `python -m pip install -e ".[excel]"` 安装可选依赖。

准备真实数据并设置分析配置后运行：

```bash
python -m factorcode.research_engine factors   # 计算并合并因子
python -m factorcode.research_engine analysis  # IC 与分组数值分析
python -m factorcode.research_engine all       # 计算、分析与图表
```

其他命令包括 `ic`、`group`、`phase`，详见 `python -m factorcode.research_engine --help`。安装后也可使用 `factorroute` 命令。

## 数据与配置

配置集中在 `factorcode/project_config.py`。输入为每指标一张 CSV 宽表：首列日期，其余列证券代码；文件映射见 `METRIC_FILES`。运行前需明确样本期、基准、复权、币种及财务数据可得时点。仓库不包含行情，当前读取本地文件，赛事接口尚未接入。

| 环境变量 | 用途 |
|---|---|
| `FACTORROUTE_HOME` | 数据与输出工作区，默认项目根目录 |
| `FC_WIDE_CSV_DIR` | 输入目录，默认 `data/raw/` |
| `FC_ANALYSIS_START` / `FC_ANALYSIS_END` | 分析起止日期 |
| `FC_BENCHMARK` | 基准表中的指数列名 |

环境变量在启动 Python 前设置，`.env.example` 仅提供示例，不自动加载。

结果写入 `outputs/`：`factors/` 为单因子，`style_factors_merged.csv` 为合并表，`style_factor_correlation.csv` 为相关矩阵，`ic/` 和 `backtest/` 为分析及图表。同名输出会覆盖；实验参数、数据版本和结论保存在 `experiments/`。数据、环境和运行结果默认不纳入版本管理。

## 当前范围

现有 IC 与分组研究使用周期平均因子值及收盘价收益，Top−Bottom 收益用于因子诊断。尚未实现周五收盘快照、下一交易日开盘成交、组合现金与持仓、交易成本和订单核对，研究收益不等同于参赛策略净值。

市场、选股因子、评分权重、持股数量及风险参数尚未固定。候选因子不自动全部进入正式评分，市场适用性和完整数据链路仍需验证。
