"""周研究阶段编排及字段白名单导出；不读取账户或提交订单。"""
from pathlib import Path
from datetime import datetime, timezone
import json
import math
import platform
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
from .data.yfinance_source import download_snapshot, load_snapshot
from .strategy.weekly import compute_factors, rank_universe, select_portfolio
from .analysis.portfolio import simulate, stats, build_schedule, benchmark_nav, plot_nav

PUBLIC_COLUMNS=['date','code','company','sector','score','rank','target_weight']
LIMITATIONS=['Current constituents retrospectively: survivorship and membership lookahead bias',
    'Adjusted-price total-return approximation, fractional shares and assumed full execution',
    'No precise dividend cash, FX, tax, lot-size, halt or order-execution ledger',
    'No sector cap; earnings events not part of selection exclusion',
    '2026 review previously inspected; not pristine out-of-sample; historical drawdown is not a loss limit']

def clean_json(value):
    if isinstance(value,dict): return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean_json(v) for v in value]
    if isinstance(value,(float,np.floating)): return float(value) if math.isfinite(value) else None
    if isinstance(value,np.integer): return int(value)
    return value

def save_json(path,value):
    Path(path).write_text(json.dumps(clean_json(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def _navs(picks,market,cost):
    close,opening=market['close'],market['open']; schedule=build_schedule(picks,close.index)
    curves={}; trades=None
    if not schedule:
        return pd.DataFrame({'strategy':1.,'before_cost':1.,'SPY':1.},index=close.index),pd.DataFrame(),0
    first=min(schedule); baseline=close.index[close.index<first][-1]
    for label,fee in [('strategy',cost),('before_cost',0.)]:
        nav,transactions=simulate(close,opening,schedule,fee)
        curves[label]=pd.concat([pd.Series([1.],index=[baseline]),nav])
        if label=='strategy': trades=transactions
    curves['SPY']=benchmark_nav(market['benchmark'],first,baseline,close.index[-1],cost)
    navs=pd.DataFrame(curves); navs.index.name='date'
    if navs.isna().any().any(): raise ValueError('Missing portfolio/benchmark valuation')
    return navs,trades,len(schedule)

def normalized_stats(nav,start,end):
    previous=nav.loc[nav.index<pd.Timestamp(start)]; part=nav.loc[start:end]
    if part.empty: raise ValueError('Empty study period')
    if not previous.empty: part=pd.concat([previous.iloc[-1:],part])
    return stats(part/part.iloc[0])

def study_weights(rankings,market,config,out):
    rows=[]
    for weight in config['momentum_grid']:
        candidate=dict(config,weights={'RSTR':weight,'DASTD':1-weight})
        picks=select_portfolio(rankings,candidate)
        navs,_,_=_navs(picks,market,config['cost_rate'])
        train=normalized_stats(navs.strategy,config['train_start'],config['train_end'])
        review=normalized_stats(navs.strategy,config['review_start'],config['review_end'])
        rows.append({'momentum':weight,'low_vol':1-weight,'train_sharpe':train['sharpe_rf0'],
            'train_return':train['total_return'],'train_mdd':train['max_drawdown'],
            'review_return':review['total_return'],'review_mdd':review['max_drawdown'],
            'review_sharpe':review['sharpe_rf0'],'full_final_nav':float(navs.strategy.iloc[-1])})
    table=pd.DataFrame(rows); table.to_csv(out/'weight_comparison.csv',index=False)
    eligible=table[table.train_mdd>=-config['train_drawdown_screen']].dropna(subset=['train_sharpe'])
    if eligible.empty: raise ValueError('No grid candidate passes historical training drawdown screen')
    best=eligible.sort_values(['train_sharpe','momentum'],ascending=[False,True]).iloc[0].to_dict()
    save_json(out/'weight_study.json',{'selected_by_training_only':best,
        'config_unchanged':True,'review_previously_inspected':config['review_previously_inspected'],
        'drawdown_screen_is_exploratory_and_not_guaranteed':True})
    return table

def research(market,config,out):
    factors=compute_factors(market['close'],market['benchmark'])
    factors.to_csv(out/'factors_daily.csv',index=False)
    rankings=rank_universe(factors,market,config); picks=select_portfolio(rankings,config)
    rankings.to_csv(out/'weekly_rankings.csv',index=False); picks.to_csv(out/'weekly_picks.csv',index=False)
    latest=picks[picks.date==pd.Timestamp(config['signal_date'])].copy()
    latest.to_csv(out/'selected_internal.csv',index=False)
    navs,trades,count=_navs(picks,market,config['cost_rate'])
    navs.to_csv(out/'daily_nav.csv'); trades.to_csv(out/'rebalance_costs.csv',index=False)
    market['quality'].to_csv(out/'data_quality.csv',index=False)
    metrics={key:stats(navs[key]) for key in navs}
    public_config={key:value for key,value in config.items() if not key.startswith('_')}
    sectors=latest.groupby('sector').target_weight.sum().sort_values(ascending=False).to_dict()
    summary={'parameters':public_config,'metrics':metrics,'signal_date':config['signal_date'],
        'valuation_end':str(market['close'].index.max().date()),'rebalance_count':count,
        'execution_status':'historical simulation only; no orders; latest signal may be pending first entry',
        'calendar_check':market['calendar_status'],'warmup_skipped':rankings.attrs.get('warmup_skipped',[]),
        'sector_weights':sectors,'mean_beta_proxy':float(latest.HBETA.mean()),
        'boundary_tie_count':int(latest.boundary_tie_count.iloc[0]),
        'software':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__},
        'limitations':LIMITATIONS}
    for name,key in [('train',('train_start','train_end')),('review',('review_start','review_end'))]:
        if all(k in config for k in key):
            summary[name]={column:normalized_stats(navs[column],config[key[0]],config[key[1]]) for column in navs}
    save_json(out/'summary.json',summary)
    title=(f'S&P 500 | Top {config["holdings"]} | Momentum {config["weights"]["RSTR"]:.0%} / Low volatility {config["weights"]["DASTD"]:.0%}\n'
        f'Initial 1 -> {metrics["strategy"]["final_nav"]:.4f} | MDD {metrics["strategy"]["max_drawdown"]:.2%} | Research approximation')
    plot_nav(navs,out,title)
    return {'summary':summary,'rankings':rankings,'picks':picks}

def publish_week(result_dir,week_dir):
    result_dir,week_dir=Path(result_dir),Path(week_dir)
    names=['selected.csv','metrics.json','equity.png']
    if any((week_dir/name).exists() for name in names): raise FileExistsError('Published results already exist; preserve historical record')
    week_dir.mkdir(parents=True,exist_ok=True)
    picks=pd.read_csv(result_dir/'selected_internal.csv')
    picks.loc[:,PUBLIC_COLUMNS].to_csv(week_dir/'selected.csv',index=False)
    summary=json.loads((result_dir/'summary.json').read_text())
    # 白名单构造，禁止直接复制可能含本机或私有字段的summary。
    public={key:summary[key] for key in ['signal_date','valuation_end','metrics','rebalance_count',
        'execution_status','calendar_check','sector_weights','mean_beta_proxy','boundary_tie_count','software','limitations']}
    public['strategy']={key:summary['parameters'][key] for key in ['weights','directions','score_method','holdings','cost_rate']}
    for key in ['train','review']:
        if key in summary: public[key]=summary[key]
    save_json(week_dir/'metrics.json',public)
    shutil.copy2(result_dir/'equity.png',week_dir/'equity.png')

def run(config,stage,study=False,open_chart=False,publish=False):
    if stage not in ('download','research','all'): raise ValueError('Invalid stage')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=config['_root']/'outputs/weekly'/config['week']/stamp
    out.mkdir(parents=True,exist_ok=False)
    source=config['_snapshot']
    if stage in ('download','all'):
        source=download_snapshot(config,config['_root']/'data/yfinance'/f'snapshot_{stamp}')
        save_json(out/'download.json',{'snapshot':str(source.relative_to(config['_root']))})
        print('Downloaded snapshot:',source)
    if stage=='download': return out
    config=dict(config,snapshot=str(source.relative_to(config['_root'])))
    market=load_snapshot(source,config['signal_date']); result=research(market,config,out)
    if study: study_weights(result['rankings'],market,config,out)
    manifest=source/'manifest.json'
    if manifest.exists(): shutil.copy2(manifest,out/'input_manifest.json')
    if publish: publish_week(out,config['_root']/'experiments/weekly'/config['week'])
    value=result['summary']['metrics']['strategy']
    print(f'Initial 1 -> {value["final_nav"]:.4f}; total return {value["total_return"]:.2%}; max drawdown {value["max_drawdown"]:.2%}')
    print('Chart:',out/'equity.png')
    if open_chart:
        path=out/'equity.png'
        try:
            if sys.platform=='darwin': subprocess.run(['open',str(path)],check=True)
            elif sys.platform=='win32': __import__('os').startfile(path)
            else: subprocess.run(['xdg-open',str(path)],check=True)
        except (OSError,subprocess.SubprocessError): print('Open chart manually:',path)
    return out
