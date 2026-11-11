# 11 月 9 日实验复现

本周工作和策略结果见 [周报](report.md)。在仓库根目录运行，`outputs/weekly/2026-11-09/` 下的冻结数据和结果由本地归档提供。

## 数据、权重搜索与选股

- `free_dynamic_pool.py` 读取 `free-pit-pilot/` 下的历史成员、身份映射和补价。`fetch_coverage_gaps.py` 下载后续报价缺口，`audit_free_data_range.py` 读取其 `yahoo/` 子目录并输出覆盖检查。
- `run_wide_spy_grid.py` 读取冻结的 `free-data-audit/` 和 `data/yfinance/extended_2015_to_20261030/`，调用 `run_spy_grid.py` 与 `window_grid.c`。同一新建结果目录可用原命令续跑。
- `compare_nov06_shortlist.py` 读取 `spy-grid-wide/` 的冻结权重及 `spy-candidate-pilot/` 和 `capped-buffer-week-review/` 的旧名单。先用 `download` 建立报价快照，再对同一目录运行 `compare`。

```sh
rtk proxy .venv-weekly/bin/python experiments/weekly/2026-11-09/run_wide_spy_grid.py outputs/weekly/2026-11-09/reproduce-wide
rtk proxy .venv-weekly/bin/python experiments/weekly/2026-11-09/compare_nov06_shortlist.py download outputs/weekly/2026-11-09/reproduce-nov06
rtk proxy .venv-weekly/bin/python experiments/weekly/2026-11-09/compare_nov06_shortlist.py compare outputs/weekly/2026-11-09/reproduce-nov06
```

## 八组对照

在项目根目录运行。下面使用新的示例目录；每次复现都必须更换唯一运行名，不能复用已经存在的目录。本次真实运行目录为`outputs/weekly/2026-11-09/20261105-eight-way`。

```sh
rtk proxy .venv-weekly/bin/python experiments/weekly/2026-11-09/compare_strategies.py freeze --run-dir outputs/weekly/2026-11-09/reproduce-unique-run
rtk proxy .venv-weekly/bin/python experiments/weekly/2026-11-09/compare_strategies.py download --run-dir outputs/weekly/2026-11-09/reproduce-unique-run
rtk proxy .venv-weekly/bin/python experiments/weekly/2026-11-09/compare_strategies.py evaluate --run-dir outputs/weekly/2026-11-09/reproduce-unique-run
```

复现不带`--publish`，结果保存在新运行目录；本周已发布的精选文件拒绝覆盖。
