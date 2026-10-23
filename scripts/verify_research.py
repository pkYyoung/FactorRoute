"""使用独立工作区和合成数据验证研究链路，不调用外部数据服务。"""
from pathlib import Path
import os
import subprocess
import sys
import tempfile

SCENARIO = r"""
import importlib
import pkgutil
from pathlib import Path
import numpy as np
import pandas as pd
import factorcode
from factorcode import project_config as c
from factorcode.data import DataIO, FactorPipeline, BARRA_FACTOR_CLASSES, ALL_FACTOR_CLASSES, FACTOR_CLASS_BY_NAME
from factorcode.analysis.ic import ICAnalysis
from factorcode.analysis.group import GroupAnalysis

for module in pkgutil.walk_packages(factorcode.__path__, factorcode.__name__ + "."):
    importlib.import_module(module.name)
assert len(BARRA_FACTOR_CLASSES) == len(ALL_FACTOR_CLASSES) == len(FACTOR_CLASS_BY_NAME) == 16
assert set(c.RUN_TARGETS) == set(FACTOR_CLASS_BY_NAME)
assert len(c.LEVEL1_WEIGHTED) == 16
raw = c.DATA_RAW
raw.mkdir(parents=True)
dates = pd.bdate_range("2024-01-01", periods=80)
codes = ["00005.HK", "AAPL"] + [f"TEST{i}" for i in range(18)]
rng = np.random.default_rng(42)
prices = 100 * np.exp(np.cumsum(rng.normal(0, .01, (80, 20)), axis=0))
caps = rng.uniform(1e8, 1e10, (80, 20))
for name, values in [("close_daily.csv", prices), ("mktcap_total_daily.csv", caps)]:
    frame = pd.DataFrame(values, index=dates, columns=codes)
    frame.index.name = "date"
    frame.to_csv(raw / name)
io = DataIO()
pipeline = FactorPipeline(io)
size = pipeline.compute_one("Size")
assert len(size) == 1600 and set(size.code) == set(codes)
expected = pd.DataFrame(caps, index=dates, columns=codes).rename_axis("date").reset_index().melt(id_vars="date", var_name="code", value_name="cap")
check = size.merge(expected, on=["date", "code"])
assert np.allclose(check.LNCAP, np.log(check.cap))
merged = size.rename(columns={"Size": "SIZE"})[["date", "code", "SIZE"]]
c.DATA_OUT.mkdir(exist_ok=True)
merged.to_csv(c.FACTORS_MERGED, index=False)
c.FACTOR_COLUMNS = ["SIZE"]
ICAnalysis(io, pipeline, c).run()
GroupAnalysis(io, pipeline, c).run()
for file in [c.DATA_IC / "ic_series.csv", c.DATA_IC / "ic_stats.csv", c.DATA_BACKTEST / "backtest_premium_daily.csv", c.DATA_BACKTEST / "backtest_stats.csv"]:
    assert file.exists() and not pd.read_csv(file).empty, file
assert c.TEMPORAL_EXPORT_FREQS == ("month",)
assert c.WORKSPACE_ROOT == Path.cwd()
print("PASS: core package imports; symbols; Size formula; IC; Group; output CSVs")
"""


def main():
    with tempfile.TemporaryDirectory(prefix="factorroute-check-") as directory:
        env = dict(os.environ, FACTORROUTE_HOME=directory, MPLCONFIGDIR=directory + "/mpl", MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
        for name in ["FC_WIDE_CSV_DIR", "FC_ANALYSIS_START", "FC_ANALYSIS_END", "FC_RUN_STYLES"]:
            env.pop(name, None)
        for module in ["factorcode.research_engine", "factorcode.analysis.ic", "factorcode.analysis.group"]:
            subprocess.run([sys.executable, "-m", module, "--help"], cwd=directory, env=env, check=True, stdout=subprocess.DEVNULL, timeout=90)
        subprocess.run([sys.executable, "-c", SCENARIO], cwd=directory, env=env, check=True, timeout=90)
    print("PASS: independent workspace and CLI entry points")


if __name__ == "__main__":
    main()
