"""周配置与根目录相对路径；不在公开配置中保存账户信息。"""
from pathlib import Path
import json
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def load_config(path):
    config = json.loads(Path(path).expanduser().resolve().read_text(encoding='utf-8'))
    weights = config.get('weights', {})
    if set(weights) != {'RSTR', 'DASTD'} or not all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0
        for x in weights.values()
    ) or not math.isclose(sum(weights.values()), 1, abs_tol=1e-9):
        raise ValueError('weights must be finite, nonnegative and sum to 1')
    required = ('week', 'signal_date', 'snapshot', 'download_start', 'download_end_exclusive',
                'analysis_start', 'directions', 'score_method', 'holdings', 'min_eligible',
                'cost_rate', 'adv_window', 'adv_min_usd')
    missing = [key for key in required if key not in config]
    if missing: raise ValueError(f'Missing configuration: {missing}')
    for key in ('holdings', 'min_eligible', 'adv_window'):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if config['min_eligible'] < config['holdings']: raise ValueError('min_eligible < holdings')
    for key in ('cost_rate', 'adv_min_usd'):
        if not isinstance(config[key], (float, int)) or not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f'Invalid {key}')
    if config['cost_rate'] >= .1: raise ValueError('cost_rate must be below 0.1')
    if config['score_method'] not in ('zscore', 'rank'): raise ValueError('Invalid score_method')
    if config['directions'] != {'RSTR': 1, 'DASTD': -1}: raise ValueError('Invalid factor directions')
    start, signal, end, week = map(pd.Timestamp, [config['download_start'], config['signal_date'],
        config['download_end_exclusive'], config['week']])
    if any(pd.isna(d) for d in (start, signal, end, week)) or not start < signal < end or not signal < week or week.weekday() != 0:
        raise ValueError('Invalid signal/download/week dates')
    if signal.to_period('W-FRI').end_time.normalize() >= week: raise ValueError('Signal week must precede holding week')
    snapshot = Path(config['snapshot'])
    if snapshot.is_absolute() or '..' in snapshot.parts: raise ValueError('snapshot must be root-relative')
    for key in ('reported_on', 'analysis_start', 'train_start', 'train_end', 'review_start', 'review_end'):
        if key in config: pd.Timestamp(config[key])
    config['_root'] = ROOT
    config['_snapshot'] = ROOT / snapshot
    return config
