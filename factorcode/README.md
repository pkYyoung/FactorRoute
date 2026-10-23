# factorcode

FactorRoute 的因子计算与分析包。安装、运行、数据格式和输出位置见[项目 README](../README.md)。

## 模块职责

| 文件 | 职责 |
|---|---|
| `project_config.py` | 指标映射、因子参数、分析与绘图配置 |
| `research_engine.py` | ResearchEngine 统一编排与命令行入口 |
| `data/io.py` | DataIO：读数、宽长表转换、缺失处理、财务可得日对齐 |
| `data/factors.py` | BarraFactor 基类、16 个因子类、描述子与类清单 |
| `data/pipeline.py` | FactorPipeline：计算、保存、合并、预处理及共享数学工具 |
| `analysis/ic.py` | IC 序列、统计、时间切片和曲线 |
| `analysis/group.py` | 分组收益、绩效、阶段分析与图表 |
| `analysis/metrics.py` | IC 与收益、波动、夏普、回撤、Calmar、胜率公式 |
| `analysis/common.py` | 周期变换、收益复利、表格转换及配色 |
| `configs/phases/phases_example.json` | 阶段列表模板；使用时填写 label/start/end |

## 候选因子

| 简称 | 风格 | 类名 |
|---|---|---|
| SIZE | 规模 | SizeFactor |
| BETA | 市场敏感度 | BetaFactor |
| LIQUID | 流动性 | LiquidityFactor |
| MOM | 动量 | MomentumFactor |
| RESVOL | 残差波动率 | ResidualVolatilityFactor |
| MIDCAP | 中等市值 | MidCapitalizationFactor |
| PROFIT | 盈利能力 | ProfitabilityFactor |
| LEVERAGE | 杠杆 | LeverageFactor |
| EQLTY | 盈利质量 | EarningsQualityFactor |
| INVQ | 投资质量 | InvestmentQualityFactor |
| GROWTH | 成长 | GrowthFactor |
| EVAR | 盈利波动 | EarningsVariabilityFactor |
| BTOP | 账面市值比 | BookToPriceFactor |
| EYLD | 盈利收益率 | EarningsYieldFactor |
| DIVYLD | 股息率 | DividendYieldFactor |
| LREV | 长期反转 | LongTermReversalFactor |

## 开发约定

- 批量清单由 `BARRA_FACTOR_CLASSES` 决定；`FACTOR_COLUMNS` 选择下游分析列；`FC_RUN_STYLES` 选择描述子目标，不筛选批量类。
- `LEVEL1_WEIGHTED` 配置风格内部的描述子权重，不是策略评分权重。因子合并只拼接列，不自动产生综合得分。
- IC 与 Group 分别调用共享预处理；默认 MAD 缩尾后标准化，避免对已预处理表重复处理。
- 部分参数在导入时绑定，参数实验使用独立进程与工作区。MidCapitalization 依赖 Size 输出，批量顺序具有意义。
- 财报对齐保留预计披露日和报告期后 60 日的回退逻辑，真实数据接入时需核验。年化指标采用 252 个交易日约定。
- 合并表相关矩阵是分析期长表 Pearson 相关。CSV 重绘使用落盘精度，可能与内存结果绘图略有差异。
