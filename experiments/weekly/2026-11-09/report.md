# FactorRoute 周报 · 2026-11-09 计划持仓周

信号日和行情截止日均为 2026-11-06（美股收盘）。Alpha 55 五因子策略的计划执行日为 2026-11-10，实际成交和账户净值另行记录。

## 本周更新

- 完成 10 月 30 日信号的八组对照，检查评分方法、持股数和行业上限。11 月 2 日开盘到 11 月 4 日收盘，八组都落后于同期 SPY。
- 改用信号日生效的标普 500 名单，核对历史证券身份并补齐免费日线。连续通过报价和因子历史门槛的回测期为 2023 年 3 月 20 日开盘至 2026 年 10 月 30 日收盘，包含 177 个滚动及 14 个不重叠的 13 周窗口。
- 在 9 个日线指标上搜索 3,504,105 套去重配方。全因子组合按 5 个百分点搜索，最多三个因子的组合按 1 个百分点搜索。只用截至 2025 年底的窗口排榜，锁定前 20 名后再复核 2026 年。

## 策略与本周名单

**Alpha 55 五因子策略**（简称 **Alpha 55**）按当周合格股票的指标标准分加权，使用 **55% Alpha、20% 原动量 RSTR、15% 低残差波动、5% 低市场 Beta、5% 同行业 21 日相对动量**。这套配方在冻结训练榜单上排名第 6。每周末选 10 只等权股票，仍合格且排前 15 名的老仓在行业名额内优先保留，调仓时同一行业最多 3 只。

11 月 6 日信号从前一周的 70/30 归档名单出发，保留 4 只、替换 6 只。能源和医疗各占目标组合 30%。完整目标见[本周精选名单](selected.csv)。

## 回测与验证

2026 年 1 月 5 日至 10 月 30 日，按买卖成交额计 10bps，13 周窗口分别从现金开始。

| 方案 | 三个不重叠窗口的净收益中位数 | 不重叠窗口最深收盘回撤 | 不重叠窗口跑赢 SPY | 滚动窗口跑赢 SPY |
|---|---:|---:|---:|---:|
| SPY | +2.94% | 8.89% | — | — |
| 70/30 基线 | +8.19% | 13.77% | 2/3 | 18/31 |
| Alpha 55 | +14.94% | 9.35% | 3/3 | 29/31 |

15bps 下，Alpha 55 仍在三个不重叠窗口里逐一跑赢 SPY 和 70/30，回撤也都浅于 70/30。早期排榜段，Alpha 55 相对 70/30 的不重叠窗口收益差中位数为 +0.23 个百分点。2026 年数据此前已查看过，本轮结果属于历史复核。滚动窗口共享交易日，行业和部分退出股仍有覆盖缺口。

11 月 2 日开盘至 11 月 6 日收盘的另一次模拟中，70/30 为 −10.05%，Alpha 55 为 −7.76%，SPY 为 −3.28%，均包含 10bps 初始买入费。11 月 6 日名单与这段持仓使用不同信号日。

## 复现

在仓库根目录使用 `.venv-weekly`。冻结行情和过程产物位于本地 `outputs/weekly/2026-11-09/`，以下命令使用独立输出目录。

### 八组对照

[冻结配置](research/strategy_comparison/config.json) · [名单](research/strategy_comparison/selected.csv) · [指标](research/strategy_comparison/metrics.json) · [净值图](research/strategy_comparison/equity.png)

每次换一个尚不存在的运行目录。历史运行在本地 `20261105-eight-way/`，原实验报告在 `archive/eight-way-report.md`。复现默认写入运行目录，`--publish` 会拒绝覆盖已归档文件。

```sh
module=experiments.weekly.2026-11-09.research
out=outputs/weekly/2026-11-09/reproduce-eight-way
rtk proxy .venv-weekly/bin/python -m "$module.strategy_comparison.verify"
rtk proxy .venv-weekly/bin/python -m "$module.strategy_comparison.run" freeze --run-dir "$out"
rtk proxy .venv-weekly/bin/python -m "$module.strategy_comparison.run" download --run-dir "$out"
rtk proxy .venv-weekly/bin/python -m "$module.strategy_comparison.run" evaluate --run-dir "$out"
```

### 动态成分股验收与权重搜索

动态成员入口读取 `free-pit-pilot/` 的历史成员、身份映射和补价。补价入口生成 `yahoo/`，覆盖验收读取该目录并在同一运行目录写出覆盖表。离线验收可先复制冻结的 `free-data-audit/yahoo/` 到新目录。

全量权重搜索读取冻结的 `free-data-audit/` 和 `data/yfinance/extended_2015_to_20261030/`。冻结榜单见本地 `spy-grid-wide/frozen_top20.csv`。全量任务可在同一运行目录按原命令续跑，小批量验收需用独立目录并加 `--limit 12 --workers 2 --part-size 6`。

```sh
module=experiments.weekly.2026-11-09.research
base=outputs/weekly/2026-11-09
rtk proxy .venv-weekly/bin/python -m "$module.data_coverage.run" --out "$base/reproduce-members"
rtk proxy .venv-weekly/bin/python -m "$module.data_coverage.download" "$base/reproduce-coverage"
rtk proxy .venv-weekly/bin/python -m "$module.data_coverage.audit" "$base/reproduce-coverage"
rtk proxy .venv-weekly/bin/python -m "$module.weight_search.run" "$base/reproduce-wide"
```

### 11 月 6 日名单

入口读取 `spy-grid-wide/` 的冻结权重，以及 `spy-candidate-pilot/` 和 `capped-buffer-week-review/` 的旧名单。离线复算使用原报价快照，先把本地 `nov06-strategy-comparison/` 的 `week_quotes.csv`、`okta.csv` 和 `download.json` 复制到新目录，再运行 `compare`。重新下载时在新目录先运行同一入口的 `download` 子命令。

```sh
module=experiments.weekly.2026-11-09.research
out=outputs/weekly/2026-11-09/reproduce-nov06
rtk proxy .venv-weekly/bin/python -m "$module.weekly_selection.run" compare "$out"
```

在输出 `comparison_lists.csv` 中取信号日为 2026-11-06、配方为 `rank_06_switch_from_baseline` 的行。按 `score_rank` 升序与精选名单对照，10 只股票各 10%。精选名单中的 `rank` 保留缓冲选股前的全池评分排名。
