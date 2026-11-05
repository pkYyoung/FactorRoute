# 8组实验复现

## 复现命令

在项目根目录运行。下面使用新的示例目录；每次复现都必须更换唯一运行名，不能复用已经存在的目录。本次真实运行目录为`outputs/weekly/2026-11-09/20261105-eight-way`。

```sh
rtk proxy .venv-weekly/bin/python experiments/weekly/2026-11-09/compare_strategies.py freeze --run-dir outputs/weekly/2026-11-09/reproduce-unique-run
rtk proxy .venv-weekly/bin/python experiments/weekly/2026-11-09/compare_strategies.py download --run-dir outputs/weekly/2026-11-09/reproduce-unique-run
rtk proxy .venv-weekly/bin/python experiments/weekly/2026-11-09/compare_strategies.py evaluate --run-dir outputs/weekly/2026-11-09/reproduce-unique-run
```

复现不带`--publish`，结果保存在新运行目录；本周已发布的精选文件拒绝覆盖。
