# Weekly Research Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 在原仓库上整理可复现的周频研究程序，公开本周脱敏成果，验证后清理冗余，并正常推送main。

**Architecture:** 原因子与分析模块保持不变，新增下载适配、周策略、账户模拟及薄命令入口。每周使用独立JSON配置，完整输入/输出只在本地，精选成果按计划持仓周保存。先迁移并回归对照，再清理，禁止让新流程依赖即将删除的旧脚本。

**Tech Stack:** Python 3.10–3.12；现有numpy 2.2.6、pandas 2.3.0、scipy 1.15.3、matplotlib 3.10.3；可选yfinance 1.7.0、exchange-calendars 4.13.2；JSON/CSV/Markdown/PNG；unittest合成验证。

## Global Constraints

- 共用代码按功能组织，每周配置、报告和精选结果按计划持仓周归档，不每周复制程序。
- 原有因子公式、IC 和 Top−Bottom 入口保持不变。
- 账户资金、参考股数和下单预算留在本地；不上传原始行情、环境和缓存。
- 保留原有Databento忽略规则；Algoseek、Databento及其他原始项目文件不属于本次删除范围。
- 按“清点—迁移—验证—删除冗余”的顺序实施。
- 正常提交并推送 `main`；不创建 PR，不使用强制推送，不覆盖队友的新提交。
- 不新增策略优化目标，不扩大下载区间，不重选本周权重，不下单。
- shell命令均按本机RTK约定加前缀；以下命令从仓库根执行。
- 代码步骤所需的旧源码必须在清理前读取；完成迁移后不再import、exec或依赖 `data/yfinance/*.py`。

---

## 文件映射和接口约定

| 文件 | 单一职责 |
|---|---|
| `factorcode/weekly_config.py` | 读取/校验周配置和解析根目录相对路径 |
| `factorcode/data/yfinance_source.py` | 下载、快照读取、质量和供应商口径 |
| `factorcode/strategy/__init__.py` | 新包标记 |
| `factorcode/strategy/weekly.py` | 调用原因子、公司资格筛选、评分和目标组合 |
| `factorcode/analysis/portfolio.py` | 日频调仓/估值、指标、SPY对照和画图 |
| `factorcode/weekly_workflow.py` | 串联阶段、运行目录、权重研究和公开导出 |
| `scripts/run_weekly.py` | 薄CLI，支持从任意工作目录按绝对配置路径运行 |
| `scripts/verify_weekly.py` | 标准库unittest的合成边界验证 |
| `experiments/weekly/2026-11-02/` | 本周公开配置、报告、名单、归一化指标和图 |
| `docs/yfinance-audit.md` | 从旧审计精简的证据和限制 |
| `README.md`、`factorcode/README.md`、`experiments/weekly/README.md` | 团队阅读路径与复现说明 |
| `.gitignore`、`pyproject.toml` | 数据排除及可选下载依赖 |

约定 `MarketData` 是字典，键为 `close`、`open`、`volume`、`dollar`、`benchmark`、`universe`、`quality`。前四项是按美股交易日索引、证券代码为列的宽表；close/open为调整研究口径，dollar为供应商Close×Volume代理。benchmark为SPY原始单层字段表；universe包含 `code/company/sector/issuer_id`；quality为逐股检查记录。

统一因子表列 `date/code/RSTR/DASTD/HBETA`；统一候选表列 `date/code/company/sector/issuer_id/RSTR/DASTD/HBETA/z_RSTR/z_DASTD/score/rank/target_weight`。旧脚本里残留的rank、score和20只target_weight不直接搬入新公开结果。

每个任务的测试都加入同一 `scripts/verify_weekly.py`，测试类置于文件末尾的 `if __name__ == '__main__'` 之前，通过 `unittest` 运行，不为本次整理引入pytest。

### Task 1: 固定基准、配置边界与依赖隔离

**Files:** Create `factorcode/weekly_config.py`、`scripts/verify_weekly.py`、`experiments/weekly/2026-11-02/config.json`；Modify `.gitignore`、`pyproject.toml`。本地基准写入 `outputs/weekly/migration/baseline/`，不提交。

**Interfaces:** `load_config(path: Path) -> dict` 返回校验后的配置，内部 `_root: Path`、`_snapshot: Path` 仅供运行，绝不公开导出。各模块消费下面的显式配置结构。

- [x] **Step 1: 保存基准，不运行旧脚本重新覆盖结果。** 使用完整代码：

```python
from pathlib import Path
import hashlib, json, shutil
r = Path.cwd()
src = r / 'data/yfinance/sp500_20261101T051000095467Z'
old = r / 'data/yfinance/backtests/20261101_131733_273805'
dst = r / 'outputs/weekly/migration/baseline'
dst.mkdir(parents=True, exist_ok=False)
for name in ('daily_nav.csv', 'latest_picks.csv', 'summary.json'):
    shutil.copy2(old / name, dst / name)
checks = {str(p.relative_to(src)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in (src / 'raw').glob('*.csv')}
(dst / 'input_sha256.json').write_text(json.dumps(checks, indent=2))
```

同时将最新快照中的私有计划、原权重比较及原最终验证保留在原处；此时不移动或删除任何旧输入。

- [x] **Step 2: 写失败的配置测试。** 新建测试入口并执行：

```python
import json, tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from factorcode.weekly_config import load_config

class ConfigTests(unittest.TestCase):
    def test_invalid_weight_fails_before_reading_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'week.json'
            p.write_text(json.dumps({'weights': {'RSTR': .7, 'DASTD': .4}}))
            with self.assertRaisesRegex(ValueError, 'weights'):
                load_config(p)

if __name__ == '__main__':
    unittest.main()
```

Run `rtk proxy .venv/bin/python scripts/verify_weekly.py ConfigTests -v`。实现前预期模块不存在；实现后预期PASS。

- [x] **Step 3: 添加完整配置及读取实现。** 配置初值如下，不保存账户金额和股数：

```json
{
  "week": "2026-11-02",
  "signal_date": "2026-10-30",
  "reported_on": "2026-11-01",
  "snapshot": "data/yfinance/sp500_20261101T051000095467Z",
  "download_start": "2023-01-01",
  "download_end_exclusive": "2026-10-31",
  "analysis_start": "2024-01-01",
  "benchmark": "SPY",
  "universe": "sp500_current_snapshot",
  "universe_url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
  "weights": {"RSTR": 0.7, "DASTD": 0.3},
  "directions": {"RSTR": 1, "DASTD": -1},
  "score_method": "zscore",
  "holdings": 10,
  "min_eligible": 100,
  "adv_window": 20,
  "adv_min_usd": 10000000,
  "cost_rate": 0.001,
  "train_start": "2024-01-01",
  "train_end": "2025-12-31",
  "review_start": "2026-01-01",
  "review_end": "2026-10-30",
  "momentum_grid": [0, 0.25, 0.5, 0.7, 0.75, 1],
  "train_drawdown_screen": 0.25,
  "review_previously_inspected": true,
  "historical_membership": false
}
```

`weekly_config.py` 的完整基础实现：

```python
from pathlib import Path
import json, math
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]

def load_config(path):
    p = Path(path).expanduser().resolve()
    c = json.loads(p.read_text(encoding='utf-8'))
    weights = c.get('weights', {})
    if set(weights) != {'RSTR', 'DASTD'} or not all(
        isinstance(x, (int, float)) and math.isfinite(x) and x >= 0
        for x in weights.values()
    ) or not math.isclose(sum(weights.values()), 1, abs_tol=1e-9):
        raise ValueError('weights must be nonnegative finite numbers summing to 1')
    required = ('week', 'signal_date', 'snapshot', 'download_start',
                'download_end_exclusive', 'analysis_start', 'directions',
                'score_method', 'holdings', 'min_eligible', 'cost_rate')
    if any(k not in c for k in required):
        raise ValueError('missing weekly configuration fields')
    if not isinstance(c['holdings'], int) or c['holdings'] <= 0:
        raise ValueError('holdings must be a positive integer')
    if c['score_method'] not in ('zscore', 'rank'):
        raise ValueError('score_method must be zscore or rank')
    if c['directions'] != {'RSTR': 1, 'DASTD': -1}:
        raise ValueError('this strategy requires RSTR positive and DASTD negative')
    if not math.isfinite(c['cost_rate']) or not 0 <= c['cost_rate'] < .1:
        raise ValueError('invalid cost_rate')
    start, signal, end, week = map(pd.Timestamp, [c['download_start'],
        c['signal_date'], c['download_end_exclusive'], c['week']])
    if not start < signal < end or not signal < week or week.weekday() != 0:
        raise ValueError('invalid signal/download/week dates')
    c['_root'] = ROOT
    c['_snapshot'] = ROOT / c['snapshot']
    return c
```

- [x] **Step 4: 增加可选依赖及忽略规则，保留原规则。** `[project.optional-dependencies]` 新增 `yfinance = ["yfinance==1.7.0", "exchange-calendars==4.13.2"]`；不更改基础四个包版本。忽略 `/data/yfinance/`、`/data/algoseek/` 和 `/.venv-*/`；现有 `/outputs/*` 已覆盖完整结果。数据目录说明放在docs，不通过否定规则意外暴露数据。

新建隔离环境验证命令：`rtk proxy .venv/bin/python -m venv .venv-weekly`，再用 `.venv-weekly/bin/python -m pip install -e '.[yfinance]'` 安装。只在需要联网的这一步申请相应执行权限，不升级原 `.venv` 或全局包。

- [x] **Step 5: 运行配置测试和基准文件完整性检查。** 预期配置测试PASS，基准nav终值约2.7877134188，名单为PSX/MPC/VLO/RVTY/NTAP/APA/CRL/TGT/EXPD/KEYS。先不提交不完整的功能；在Task 4完成统一入口后共同提交。

### Task 2: 迁移日频账户模拟和绘图

**Files:** Create `factorcode/analysis/portfolio.py`；extend `scripts/verify_weekly.py`。

**Interfaces:** 保留 `simulate(close, op, schedule, cost) -> (pd.Series, pd.DataFrame)`、`stats(nav) -> dict`。新增 `build_schedule(picks, dates) -> dict`、`benchmark_nav(benchmark, first, baseline, last, cost) -> pd.Series`、`plot_nav(navs, out, title) -> Path`。schedule为交易日期到代码索引目标权重Series的字典。

- [x] **Step 1: 增加资金和日期测试。** 测试代码：

```python
from factorcode.analysis.portfolio import simulate, stats, build_schedule
import numpy as np
import pandas as pd

class PortfolioTests(unittest.TestCase):
    def test_flat_prices_charge_only_actual_entry(self):
        dates = pd.bdate_range('2024-01-01', periods=4)
        prices = pd.DataFrame({'A': 100., 'B': 100.}, index=dates)
        w = pd.Series({'A': .5, 'B': .5})
        nav, trades = simulate(prices, prices, {dates[0]: w, dates[2]: w}, .001)
        self.assertTrue(np.allclose(nav, 1 / 1.001))
        self.assertAlmostEqual(trades.cost.iloc[1], 0.)

    def test_last_friday_has_no_fabricated_monday(self):
        dates = pd.to_datetime(['2026-10-29', '2026-10-30'])
        picks = pd.DataFrame({'date': [dates[-1]], 'code': ['A'], 'target_weight': [1.]})
        self.assertEqual(build_schedule(picks, dates), {})

    def test_drawdown_includes_initial_capital(self):
        nav = pd.Series([1., .8, 1.], index=pd.bdate_range('2024-01-01', periods=3))
        self.assertAlmostEqual(stats(nav)['max_drawdown'], -.2)
```

Run `rtk proxy .venv/bin/python scripts/verify_weekly.py PortfolioTests -v`，预期先因模块缺失失败。

- [x] **Step 2: 原样迁移已经验证的simulate/stats函数。** 使用下面的完整迁移命令，不迁移旧main、report或硬编码plot：

```python
from pathlib import Path
import ast
source = Path('data/yfinance/sp500_equity_backtest.py').read_text()
tree = ast.parse(source)
functions = [ast.get_source_segment(source, node) for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in ('simulate', 'stats')]
assert len(functions) == 2
header = 'import numpy as np\nimport pandas as pd\nfrom . import metrics as m\n\n'
Path('factorcode/analysis/portfolio.py').write_text(header + '\n\n'.join(functions) + '\n')
```

将运行时关键assert转成明确的ValueError/RuntimeError，条件和资金口径不变；不能在缺失持仓报价时继续按0处理。

- [x] **Step 3: 添加无未来交易和同口径基准实现。**

```python
def build_schedule(picks, dates):
    result = {}
    for signal, group in picks.groupby('date'):
        future = dates[dates > signal]
        if len(future):
            result[future[0]] = group.set_index('code').target_weight
    return result

def benchmark_nav(benchmark, first, baseline, last, cost):
    opening = benchmark.Open * benchmark['Adj Close'] / benchmark.Close
    nav = benchmark.loc[first:last, 'Adj Close'] / opening.loc[first] / (1 + cost)
    return pd.concat([pd.Series([1.], index=[baseline]), nav])

def plot_nav(navs, out, title):
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, layout='constrained')
    for name, series in navs.items():
        ax.plot(series.index, series, label=name)
        bx.plot(series.index, series / series.cummax() - 1)
    ax.set_title(title, loc='left'); ax.set_ylabel('Initial 1 unit'); ax.legend()
    bx.set_ylabel('Drawdown'); bx.yaxis.set_major_formatter(PercentFormatter(1))
    for item in (ax, bx): item.grid(alpha=.2)
    path = Path(out) / 'equity.png'
    fig.savefig(path, dpi=160); plt.close(fig)
    return path
```

处理没有schedule的情形：保留候选导出，账户净值保持1，指标中标明没有已执行期间，不能调用min空字典或捏造收益。零波动收益的夏普和Calmar使用null，不写NaN/Infinity到公开JSON。

- [x] **Step 4: 运行三项测试及截断测试。** 对合成路径截断后再次simulate，要求前缀净值逐点一致；这是资金/时间正确性检查，不只是输出文件存在检查。

- [x] **Step 5: 保留未提交变更，等待Task 4完成整套功能验证后一起提交。**

### Task 3: 数据适配与原始因子调用、统一评分

**Files:** Create `factorcode/data/yfinance_source.py`、`factorcode/strategy/__init__.py`、`factorcode/strategy/weekly.py`；extend `scripts/verify_weekly.py`。

**Interfaces:** 数据模块 `download_snapshot(config, dest: Path) -> Path`、`load_snapshot(path: Path, signal_date: str) -> dict`。策略模块 `compute_factors(close, benchmark) -> DataFrame`、`rank_universe(factors, market, config) -> DataFrame`、`select_portfolio(rankings, config) -> DataFrame`。

- [x] **Step 1: 增加评分方向、权重确实改变选股、公司去重测试。** 合成三家公司A/B/C，动量标准分为2/1/0，波动标准分为2/0/−1。纯动量应选A，纯低波应选C，50/50应按同分代码顺序确定结果。另加入与A同issuer_id的A2，日均成交额小于A，要求仅保留A。测试通过真实公共函数，不在测试里重复实现评分。

```python
from factorcode.strategy.weekly import select_portfolio

class ScoringTests(unittest.TestCase):
    def test_weights_change_names_and_output_has_new_rank(self):
        frame = pd.DataFrame({'date': pd.to_datetime(['2026-10-30'] * 3),
            'code': ['A', 'B', 'C'], 'issuer_id': ['1', '2', '3'],
            'z_RSTR': [2., 1., 0.], 'z_DASTD': [2., 0., -1.]})
        config = {'weights': {'RSTR': 1., 'DASTD': 0.},
                  'score_method': 'zscore', 'holdings': 1}
        a = select_portfolio(frame, config)
        config['weights'] = {'RSTR': 0., 'DASTD': 1.}
        b = select_portfolio(frame, config)
        self.assertEqual(a.code.tolist(), ['A'])
        self.assertEqual(b.code.tolist(), ['C'])
        self.assertEqual(a['rank'].tolist(), [1])
        self.assertEqual(a.target_weight.tolist(), [1.])
```

- [x] **Step 2: 实现下载和读取边界，迁移旧fetch的请求语义。** 使用 `sp500_pipeline_v2.py:fetch` 的20只批次、threads=2、timeout=15、keepna/actions=True、auto_adjust/repair=False。把START/END/BASE替换为config及dest参数，不保留模块级可变路径。用下面的明确请求片段：

```python
def request_batch(symbols, config):
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError('Install factorroute[yfinance] for download stage') from exc
    return yf.download(symbols, start=config['download_start'],
        end=config['download_end_exclusive'], interval='1d', actions=True,
        auto_adjust=False, repair=False, keepna=True, threads=2,
        progress=False, timeout=15, group_by='ticker')
```

`download_snapshot`创建新dest并保存名单网页、universe.csv、raw/*.csv、manifest.json。名单优先config的本地名单，未提供则沿用已验证Wikipedia/公开CSV回退；保留来源URL和获取时刻。批量失败只对最多15只做一次单只恢复，每次间隔至少1秒；失败过多停止，不大规模逐只轰炸。最终覆盖不足、SPY缺失、信号日期不完整均报错。传入的dest可与config.snapshot不同，all阶段必须继续使用此次真正下载的目录。

`load_snapshot`逐股按旧compute的检查读取raw，禁止重复日期、非正OHLC、关系异常、负量、持仓研究范围内部缺口；开头上市空段允许但保留原因。以SPY日期为研究索引，另用可用的exchange-calendars XNYS核验。可选日历库未安装时明确标记未独立核验，不把未验证写成PASS。

读股票池时issuer_id按字符串去除`.0`并补零，映射Symbol/证券名/行业到统一列。生成close/open/volume/dollar四宽表，open=Yahoo Open×Adj Close/Close；记录价格口径。每股需在信号日有有效完整OHLCV，否则排除并计入覆盖检查。

- [x] **Step 3: 直接调用原三因子，不复制公式。** 采用旧compute的只读duck IO，完整调用段为：

```python
from ..data.factors import MomentumFactor, ResidualVolatilityFactor, BetaFactor

def compute_factors(close, benchmark):
    daily = close.rename_axis('date').reset_index().melt(
        id_vars='date', var_name='code', value_name='value')
    class IO:
        def read_metric(self, name):
            if name != '收盘价': raise KeyError(name)
            return daily.copy()
        def apply_missing_rules(self, frame, name):
            return frame.dropna(subset=['value']).query('value > 0').copy()
        def load_benchmark_returns(self):
            return benchmark['Adj Close'].pct_change(fill_method=None).dropna().rename_axis('date')
    result = None
    for cls, target in ((MomentumFactor, 'RSTR'),
                        (ResidualVolatilityFactor, 'DASTD'), (BetaFactor, 'HBETA')):
        part = cls(IO()).compute(run_target=target)
        result = part if result is None else result.merge(
            part, on=['date', 'code'], how='outer', validate='one_to_one')
    return result
```

`rank_universe`沿用旧compute的完整周日期、预热、ADV20、当日量、按当日ADV公司去重，以及MAD×1.4826×3缩尾/ddof=0标准化。参数从config读取，禁止仍写死2024日期或min100。全样本缺口排除是原研究口径，需继续披露其潜在选择偏差，不额外声称无前视。

- [x] **Step 4: 实现新评分和最终排名，不继承旧rank。**

```python
def select_portfolio(rankings, config):
    parts = []
    for date, source in rankings.groupby('date', sort=True):
        x = source.copy()
        if config['score_method'] == 'zscore':
            mom, vol = x.z_RSTR, -x.z_DASTD
        else:
            mom = x.RSTR.rank(pct=True)
            vol = x.DASTD.rank(ascending=False, pct=True)
        x['score'] = config['weights']['RSTR'] * mom + config['weights']['DASTD'] * vol
        x = x.sort_values(['score', 'code'], ascending=[False, True])
        x['rank'] = range(1, len(x) + 1)
        if len(x) < config['holdings']: raise ValueError('insufficient eligible companies')
        x = x.head(config['holdings']).copy()
        if not x.issuer_id.is_unique: raise ValueError('duplicate issuer')
        x['target_weight'] = 1 / config['holdings']
        parts.append(x)
    if not parts: raise ValueError('no post-warmup signals')
    return pd.concat(parts, ignore_index=True)
```

记录最后入选分数与未入选分数的并列数；该诊断单独保存在summary，不把按代码打破并列描述为经济优势。

- [x] **Step 5: 运行评分测试、原因子前缀对照及最新快照因子计算。** 最新快照应恢复相同RSTR/DASTD/HBETA；与旧表按date/code合并，要求 `np.allclose(..., equal_nan=True)`。处理完成前不删除旧源脚本。

### Task 4: 统一工作流、命令行与离线回归

**Files:** Create `factorcode/weekly_workflow.py`、`scripts/run_weekly.py`；extend验证脚本。完整运行输出在 `outputs/weekly/<week>/<run_id>/`。

**Interfaces:** `run(config: dict, stage: str, study: bool=False, open_chart: bool=False) -> Path`；`research(market: dict, config: dict, out: Path) -> dict`；`study_weights(rankings, market, config, out) -> DataFrame`。返回summary只含研究参数和归一化指标，不含账户计划。

- [x] **Step 1: 写CLI和缺数据错误测试。** 从临时工作目录用绝对脚本和配置路径运行 `--help`，应exit0；research配置指向不存在快照时应非0，并给出“先执行download或指定快照”的错误，不访问网络。子进程环境中关闭字节码和GUI。

- [x] **Step 2: 实现薄入口。** 使用此完整入口结构：

```python
from pathlib import Path
import sys, os, argparse
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/weekly/.plot-cache'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / 'outputs/weekly/.plot-cache'))
os.environ.setdefault('MPLBACKEND', 'Agg')
from factorcode.weekly_config import load_config
from factorcode.weekly_workflow import run

def main():
    p = argparse.ArgumentParser(description='Weekly FactorRoute research')
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--stage', choices=['download', 'research', 'all'], default='research')
    p.add_argument('--study-weights', action='store_true')
    p.add_argument('--open-chart', action='store_true')
    a = p.parse_args()
    try:
        out = run(load_config(a.config), a.stage, a.study_weights, a.open_chart)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        p.exit(2, f'Error: {exc}\n')
    print(f'Results: {out}')

if __name__ == '__main__': main()
```

- [x] **Step 3: 串联阶段和结果。** 工作流创建timestamp新目录；download输出不可变快照并在结果metadata写取得位置；all直接使用新快照；research只读本地输入。研究阶段执行load_snapshot→compute_factors→rank_universe→select_portfolio→build_schedule→simulate→stats/benchmark_nav→plot_nav，保存日因子、所有周排名、最终名单、净值、成本、质量和summary。

summary写当前参数、数据版本/哈希、运行软件版本、实际起止、复核限制、公开精选列和每项验证的真实状态。生成命令在终端打印净值、累计/年化收益、回撤、图路径。图片打开仅在显式`--open-chart`时通过平台默认程序执行，失败只提示位置、不将成功研究判为失败。

权重研究使用config中的六组权重和同一套因子评分，逐组simulate，训练/复核以各段起点净值归一化。只按训练夏普在训练历史回撤约束内选候选，2026不能参与排序；结果仅输出比较表和建议，不重写config。普通周运行无该参数时完全不搜索权重。披露风险约束后设及复核期已被查看。

- [x] **Step 4: 执行有意义的回归。**

Run `rtk proxy .venv/bin/python scripts/verify_weekly.py -v`，全部PASS；再执行 `rtk proxy .venv/bin/python scripts/verify_research.py`，原框架仍PASS。执行：

```sh
rtk proxy .venv/bin/python scripts/run_weekly.py --config experiments/weekly/2026-11-02/config.json --stage research
```

与Task 1基准比较：完整NAV逐点一致（rtol=1e-10，atol=1e-12）、最新10个代码一致、每只target_weight=.1、信号2026-10-30；终值约2.7877134188、最大回撤约−0.2353618849。最新候选还没在历史中成交，不应要求其有未来收益。

从另一工作目录用绝对路径重复命令；再运行显式`--study-weights`，六组表与原表容差一致且选择70/30。保存真实的regression.json到本地migration目录。

- [x] **Step 5: 小样本验证下载依赖和安装路径。** 在新隔离环境检查安装基础包后能离线导入；没有可选包时download明确提示。安装下载extra后，对AAPL/SPY、固定短日期区间做最多一次小样本请求验证，不重新下载503只。记录网络故障为环境限制，不伪报通过。

- [x] **Step 6: 提交共用代码和配置基础。** 明确枚举Task 1–4文件，保留原未提交Databento ignore规则，然后 `rtk git commit -m "feat: unify weekly data scoring and portfolio research"`。不能把data、outputs、原私有报告加入。

### Task 5: 公开本周成果与团队文档

**Files:** Create周目录中的 `report.md/selected.csv/metrics.json/equity.png`、`experiments/weekly/README.md`、`docs/yfinance-audit.md`；Modify根及包README。工作流新增显式 `--publish`选项，仅本地导出精选结果，不代表git推送。

**Interfaces:** `publish_week(result_dir: Path, week_dir: Path) -> None` 读取研究输出，以固定白名单生成公开结果。覆盖某周已有文件须显式确认或参数，默认拒绝。

- [x] **Step 1: 写白名单导出测试。** 在候选DataFrame中人为附加 `reference_shares`、`reference_budget_hkd` 和本机路径字段；发布后输出只能有下列列，金额/股数/路径字段不能出现。

```python
PUBLIC_COLUMNS = ['date', 'code', 'company', 'sector', 'score', 'rank', 'target_weight']
public = picks.loc[picks.date == picks.date.max(), PUBLIC_COLUMNS].copy()
assert public.columns.tolist() == PUBLIC_COLUMNS
assert public['rank'].tolist() == list(range(1, len(public) + 1))
```

json只构造weights、cost假设、持股数、信号/回测区间、归一化指标、权重比较和限制，不递归复制私有summary。图表用研究title，不绘账户金额。

- [x] **Step 2: 导出本周真实结果。** 由Task 4通过回归的结果生成公开selected和图，不能直接复制含旧20只权重的candidate_plan。保留报告中：策略原理、六组比较、25%风险筛选及非保证、训练/复核口径、能源40%和财报事件50%、尚未真实成交。去掉用户资金、股数、预算与绝对路径。

公开本周事件可保留已核对的公司官网链接，但未重新验证的事件不得标注为已确认新事实。本轮不为了整理再改变名单。

- [x] **Step 3: 更新README的实际内容。** 根README明确增加：

```markdown
## 本周研究与运行

周频研究已实现yfinance历史数据适配、原因子调用、综合评分、10只组合及简化日频持仓回测；真实交易执行尚未接入。

本周冻结配置及研究说明见 experiments/weekly/2026-11-02/，全部周记录见 experiments/weekly/README.md。

安装基础研究环境：`python -m pip install -e .`；需要下载时使用 `python -m pip install -e '.[yfinance]'`。

统一入口：`python scripts/run_weekly.py --config experiments/weekly/2026-11-02/config.json --stage research --open-chart`。

首次无本地数据时先运行 `--stage download`；终端会打印新快照路径，将其填入本地周配置的snapshot后离线研究，或用 `--stage all`直接取数并研究。新下载历史可能被供应商修订，不能承诺与归档曲线逐点相同。

程序不自动下单，不将当前成分股历史研究视为无偏实盘业绩。
```

替换根README原来“评分组合及交易回测尚待开发”的过时语句，但保留原IC/Top−Bottom说明和原命令；用“简化研究模拟”准确区别于完整赛事执行。包README补新模块职责表，不重写原16因子介绍。

周索引固定包含“计划持仓周/信号日/冻结规则/本周改动/报告链接”。说明下一周只复制config并修改日期、snapshot和确认后的参数，不复制Python。审计摘要保留来源、个人使用条款未完全核验、历史覆盖小样本及价格/PIT局限，不公开原始供应商数据。

- [x] **Step 4: 检查公开导出。** 扫描tracked和拟提交文本中绝对用户目录、账户金额、股数、访问凭据字样；对命中项逐一核对用途，不输出真实秘密。配置不得有私有路径；图打开确认标题/轴/图例完整；名单10家与本地候选对应，rank是本轮最终rank。

- [x] **Step 5: 单独提交本周成果与团队说明。** 明确枚举本Task文件，`rtk git commit -m "docs: publish weekly research record for 2026-11-02"`。

### Task 6: 验证后清理及正常推送main

**Files:** 本地Create `outputs/weekly/migration/cleanup_manifest.json`、验证记录；删除仅清单中允许的旧yfinance脚本/中间物，不动其他数据源。保持不修改旧核心因子文件。

**Interfaces:** 清理器是一次性本地命令，不作为长期产品入口。每个条目记录相对路径、动作、替代位置/删除理由；执行前必须验证各路径位于data/yfinance且未超出允许集合。

- [x] **Step 1: 锁定保护集合和删除候选。** 保护：最新 `sp500_20261101T051000095467Z/raw`、universe/manifest/calendar、最终weekly_decision中的私有计划/事件来源/比较/验证；迁移基准与新的outputs；公开周目录；旧环境直到新环境验证成功。先比较最新data manifest哈希与基准，不可删除最新有效输入。

候选：`data/yfinance`根全部旧.py（已迁移无引用后）；旧 `sp500_20261029T100101025579Z` 快照；两个小样本 `run_20261029T094233662315Z`、`run_20261029T094305514345Z`；`history_probe_20261029T125046568869Z` 的可再生数据；旧 `backtests` 目录（归档精选图/指标及回归基准后）；安装、失败计算和重复运行日志；旧START_HERE及重复报告（审计结论和公开周报已迁移后）。最新快照内部的中间20只输出仅在新完整因子及排名输出保存后清理，不能通过整个目录删除顺带删掉私有计划。

- [x] **Step 2: 先生成清单和实际空间统计，再按清单执行。** 防越界代码必须包括：

```python
from pathlib import Path
import shutil
root = Path('data/yfinance').resolve()
def remove_approved(relative, approved):
    if relative not in approved: raise ValueError('not in reviewed cleanup manifest')
    target = root / relative
    if target.is_symlink(): raise ValueError('symlink deletion requires separate review')
    resolved = target.resolve()
    if root not in resolved.parents: raise ValueError('cleanup escaped yfinance root')
    if resolved.is_dir(): shutil.rmtree(resolved)
    elif resolved.exists(): resolved.unlink()
```

approved来自明确的逐路径manifest，不使用宽泛glob直接删目录。每次删除更新本地记录。新统一环境已通过安装、离线和下载小样本验证后，才删除 `.venv-audit`；若未通过则保留并说明，不为清洁牺牲复现。

- [x] **Step 3: 删除旧代码后重新跑必要检查。** 新入口不得import data脚本；运行verify_weekly、原verify_research及本周research。确认不依赖旧路径，公开结果与回归基准仍一致。统计整理前后空间，列实际保留/删除结果，不根据预计空间汇报。

- [x] **Step 4: 最终Git检查与正常同步。** 执行 `rtk git diff --check`、`rtk git status --short`、`rtk git diff --cached --stat`、`rtk git ls-files data outputs`。data中允许的旧原始.gitkeep保留，新增行情不应出现在tracked列表；不处理Algoseek/Databento未跟踪内容。

提交设计/计划状态和实际验证说明时显式枚举文档文件。随后正常fetch远端main，检查它是否仍为本地历史祖先；若队友已更新则正常整合其提交并重跑受影响检查。可用 `rtk git merge-base --is-ancestor origin/main HEAD` 判断可快进推送，失败时禁止force。

执行 `rtk git push origin main`。推送失败按真实原因报告并保留本地成果，不用强推或丢弃队友修改规避。成功后用GitHub连接器确认远端main SHA与本地HEAD一致，并读回README和本周report验证实际可见。

- [x] **Step 5: 向用户交付。** 说明：GitHub提交及周报链接、统一入口、改权重的config路径、以后每周操作、实际检查、真实清理大小、仍然只在本地的内容。明确没有改变原因子公式或完成实盘执行系统。

## 计划自检

设计的目录、职责、阶段、公开边界、保留/删除、依赖、验证与main同步均映射到上述任务。新helper只承担配置和流程编排，没有扩展到交易账户或全框架重构。所有函数名在接口块定义，测试数据和执行命令具体，旧源码迁移在删除之前。本轮保留当前数据范围与70/30研究结论，不借整理重新优化策略。


## 实施记录（2026-11-01）

六项交付已完成。共用代码提交bdb4f14，本周公开成果提交dde3ff9；设计/计划独立保留历史。原有因子公式、IO、IC、分组及指标文件相对初始提交无改动。

- 新增13项合成与边界验证全部通过；原verify_research继续通过。
- 新隔离环境固定基础版本，完成两只证券固定短区间下载和XNYS检查。
- 固定最新快照的三个原因子、最终10只代码及逐日净值与迁移前一致，NAV最大差为0；六组权重比较一致。
- 删除旧脚本后，从项目外部目录运行新入口仍通过；最新信号没有未来开盘时不虚构成交。
- 公开结果经字段白名单导出，不含账户金额、参考股数、预算或本机路径。
- 本地yfinance清理43个明确条目，按文件内容大小删除约344.22MiB，保留约53.97MiB；另清理约82.26MiB重复迁移输出。原始行情及最终私有计划保留，不动Algoseek/Databento。
- 本地基准、安装/验证日志、逐路径清理记录和完整最新输出在忽略目录outputs/weekly中，不提交。
- 正常fetch远端main后检查祖先关系，再按用户授权快进推送，不创建PR或改写历史。

原计划中的红灯步骤用于说明TDD顺序；实际配置测试先验证模块缺失，已有账户函数先原样迁移再补充合成验证。最终验收均以本次真实运行记录为准，不把历史材料中的成功声明当作新验证。
