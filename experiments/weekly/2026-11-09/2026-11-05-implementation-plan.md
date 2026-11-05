# Eight-way Strategy Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 冻结10月30日的8组名单，用11月2日至4日行情完成可复算比较。

**Architecture:** 实验脚本独立读取原归档资格池及冻结历史价格，计算两套排名和8组目标；独立估值阶段只读取已冻结名单及最新行情。所有实验代码和精选报告归档于本周目录，完整输入、审计和运行结果保存在被Git忽略的outputs目录。

**Tech Stack:** Python 3.12、现有 .venv-weekly 环境的 pandas/numpy/yfinance/matplotlib，unittest离线验证。不安装新依赖。

## Global Constraints

- 不修改 factorcode/、scripts/、上一周配置或精选结果。
- 实际信号日：2026-10-30；模拟买入日：2026-11-02；观察截止：2026-11-04美股收盘。
- 原70% RSTR / 30%低DASTD；新40%近期 / 30%中期 / 30%低波动；数量10/15；行业上限无/30%；8组等权。
- 近期21交易日相对同资格池同行其他公司的平均收益；中期63交易日绝对收益；MAD缩尾和ddof=0标准化；沿用原z_DASTD。
- floor(0.30 × N)作为行业计数上限；目标权重约束不触发期内再平衡。
- 对新组不默默缩减原资格池，缺任何必要选股输入或持仓报价须失败。
- 从现金新建仓，买入成本10bps，投资额1/(1+0.001)，期末不卖出；毛净收益与SPY并列。
- 选股不得使用10月30日之后行情；完整供应商数据不提交Git。

## 文件职责

- Create `experiments/weekly/2026-11-09/experiment_config.json`：冻结日期、窗口、权重、输入路径和8组定义。
- Create `experiments/weekly/2026-11-09/compare_strategies.py`：只读加载、评分、行业扫描、名单冻结、下载、估值、图与通俗报告；提供freeze/download/evaluate阶段，独立阶段互不覆盖已有成果。
- Create `experiments/weekly/2026-11-09/verify_experiment.py`：小规模人工数据与真实归档名单校验。
- Create `experiments/weekly/2026-11-09/report.md`、`metrics.json`、`selected.csv`、`comparison.png`：精选结果，仅研究数据和归一化绩效。
- Create `experiments/weekly/2026-11-09/progress.md`：任务、验证和审查记录。
- 完整排名、行业跳过记录、逐股价格贡献、每日净值、行情快照、质量报告及输入哈希输出到唯一运行目录 `outputs/weekly/2026-11-09/`。

### Task 1: 独立实验实现与验证

**Files:** 上述配置、Python脚本、验证脚本、progress.md。

**Interfaces:** 脚本提供明确的独立阶段CLI与输出路径；数值函数可直接导入，下载依赖延迟导入。`freeze`生成名单后，`download`读取名单并下载代码并集及SPY，`evaluate`只用该运行目录冻结输入估值；终端打印文件路径和8组汇总。

- [x] **Step 1: 写人工可算的验证，再运行确认失败。** 用21/63日有已知价格比例的面板验证窗口、同行均值排除自身，示例：

```python
peer = (sector_sum - own_return) / (sector_count - 1)
recent_relative = own_return - peer
```

验证30%行业限制在10只最多3只、15只最多4只；缺报价和无法选满时报错；价格完全不变时净收益为`1/1.001 - 1`；第一天亏损纳入最大回撤。验证构造未来异常价格不改变选股。

- [x] **Step 2: 实现配置和评分。** 从归档10月30日资格池读取原z值；价格先截断到信号日，校验日期唯一及正数，取对应63/21交易日前价格计算收益；按设计对新动量标准化。原分数与新分数分别固定，不依赖持股数。

```python
old_score = 0.7 * z_rstr - 0.3 * z_dastd
new_score = 0.4 * z_recent + 0.3 * z_medium - 0.3 * z_dastd
```

- [x] **Step 3: 实现8组扫描和冻结。** 排序`score降序/code升序`，逐个检查行业计数，选满N后停止；记录因上限跳过的股票及分数。G0逐行核对原selected.csv，G1核对原排名前15；保存配置和输入哈希，已存在冻结成果拒绝覆盖。

- [x] **Step 4: 实现行情阶段与独立估值。** 下载名单并集+SPY，日期`2026-10-30`至`2026-11-05 exclusive`，记录获取UTC时间、yfinance版本和哈希；复算时不联网。缺失交易日/非正价格/非有限值须失败。公式：

```python
adjusted_open = raw_open * adjusted_close / raw_close
gross_nav = (daily_adjusted_close / entry_adjusted_open).mean(axis=1)
net_nav = gross_nav / 1.001
drawdown = nav_with_initial_one / nav_with_initial_one.cummax() - 1
```

逐股贡献相加验证等于毛收益，不把10月30日至周一开盘跳空算入持有收益。

- [x] **Step 5: 实现报告生成和图。** 报告含8组表、SPY、逐组名单、行业分配、配对比较（数量4对、策略4对、上限4对）及通俗结论；有帮助写改善百分点，无帮助如实说明。图展示8组收益和净值，不年化，不修改正式策略。发布精选文件时拒绝覆盖，报告中的原始数据来源指向运行目录manifest。

- [x] **Step 6: 运行离线验证并自查。** `rtk proxy .venv-weekly/bin/python experiments/weekly/2026-11-09/verify_experiment.py`，预期所有实际验证通过。记录命令和测试数量，检查核心代码及原周文件哈希未变。

- [x] **Step 7: 独立审查与修正。** 审查者读取本设计、代码和测试证据，核查时点、评分、行业上限、等权、成本、报价缺失与结果归档；修复所有影响结论的问题后重跑相关验证。

### Task 2: 真实数据运行与结果交付

**Files:** 本周report.md、metrics.json、selected.csv、comparison.png、progress.md；独立运行目录的完整明细。

**Interfaces:** 使用Task 1 CLI顺序调用freeze/download/evaluate；输入输出位置写入progress.md。

- [x] **Step 1: 冻结8组名单。** 检查G0与原版完全一致，检查496家共同资格池输入完整，记录名单并集数量。只有冻结成功后才下载估值数据。
- [x] **Step 2: 下载统一行情。** 在网络受限时使用获准的联网执行；失败请求有限重试，完整覆盖才进入估值。不使用搜索摘要拼凑价格。
- [x] **Step 3: 估值与发布。** 逐组检查权重、公司唯一、行业上限、收益加总；G0约-6.15%毛收益作为交叉检查。输出全部8组结果，不只展示最好一组。
- [x] **Step 4: 视觉及数值检查。** 打开comparison.png确认图例、日期、正负号与8组齐全；用独立计算核对至少G0和最好组的期末收益，检查报告与metrics一致。
- [x] **Step 5: 最终整体审查和收尾。** 独立审查本周最终代码/报告，完成必要修正及对应测试；`rtk git diff --check`无错误。只提交本周授权文件，保留其他未跟踪文件不动，不推送或合并。向用户交付8组结果及实际限制。

## 自查结论

8组设计、数据时点、窗口、同行定义、异常处理、成本、基线复现、图与通俗报告均有对应任务。按用户指定目录原地实施，使用独立codex分支，不另建工作树、不改核心代码。本次用户已要求开始实施，执行过程不再增加审批节点。
