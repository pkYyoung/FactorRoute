"""多因子研究的统一对象入口与六个 CLI 路由。"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from . import project_config as config
from .analysis.group import GroupAnalysis
from .analysis.ic import ICAnalysis
from .data.io import DataIO
from .data.pipeline import FactorPipeline


class ResearchEngine:
    """编排 Factor → IC → Group → 图表，阶段分析保持独立。"""

    def __init__(self) -> None:
        self.io = DataIO(config.WIDE_CSV_DIR, config)
        self.pipeline = FactorPipeline(self.io, config)

    def run_factors(self):
        """运行默认 16 个 Barra 因子并在全部成功后合并。"""
        return self.pipeline.run(preprocess=False)

    def run_ic_analysis(self) -> ICAnalysis:
        return ICAnalysis(self.io, self.pipeline, config).run()

    def run_factor_group_analysis(self) -> GroupAnalysis:
        return GroupAnalysis(self.io, self.pipeline, config).run()

    def run_analysis(self) -> tuple[ICAnalysis, GroupAnalysis]:
        return self.run_ic_analysis(), self.run_factor_group_analysis()

    def run_report(self, ic: ICAnalysis, group: GroupAnalysis) -> None:
        ic.plot_curves()
        group.plot_curves()
        group.plot_extremes_vs_benchmark()
        group.plot_period_bars()
        group.plot_rrg()

    def run_phase_analysis(self, phase_config: Path | None = None):
        return GroupAnalysis(self.io, self.pipeline, config).run_phases(phase_config)

    def run_all(self) -> tuple[ICAnalysis, GroupAnalysis]:
        self.run_factors()
        ic, group = self.run_analysis()
        self.run_report(ic, group)
        return ic, group


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="多因子研究统一入口")
    parser.add_argument("command", choices=("all", "factors", "ic", "group", "analysis", "phase"))
    parser.add_argument("--config", type=Path, default=None,
                        help="阶段 JSON；仅 phase 命令使用，省略时读取 configs/phases/phases_example.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    engine = ResearchEngine()
    if args.command == "all":
        engine.run_all()
    elif args.command == "factors":
        engine.run_factors()
    elif args.command == "ic":
        engine.run_ic_analysis()
    elif args.command == "group":
        engine.run_factor_group_analysis()
    elif args.command == "analysis":
        engine.run_analysis()
    elif args.command == "phase":
        engine.run_phase_analysis(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
