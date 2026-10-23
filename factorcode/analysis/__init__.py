"""多因子研究的分析层。"""

__all__ = ["ICAnalysis", "GroupAnalysis"]


def __getattr__(name: str):
    if name == "ICAnalysis":
        from .ic import ICAnalysis

        return ICAnalysis
    if name == "GroupAnalysis":
        from .group import GroupAnalysis

        return GroupAnalysis
    raise AttributeError(name)
