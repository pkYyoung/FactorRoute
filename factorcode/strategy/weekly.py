"""原因子调用、资格过滤及截面评分。"""
import pandas as pd
from ..data.factors import MomentumFactor, ResidualVolatilityFactor, BetaFactor

def compute_factors(close, benchmark):
    daily=close.rename_axis('date').reset_index().melt(id_vars='date',var_name='code',value_name='value')
    class IO:
        def read_metric(self,name):
            if name!='收盘价': raise KeyError(name)
            return daily.copy()
        def apply_missing_rules(self,frame,name):
            return frame.dropna(subset=['value']).query('value > 0').copy()
        def load_benchmark_returns(self):
            return benchmark['Adj Close'].pct_change(fill_method=None).dropna().rename_axis('date')
    result=None
    for cls,target in ((MomentumFactor,'RSTR'),(ResidualVolatilityFactor,'DASTD'),(BetaFactor,'HBETA')):
        part=cls(IO()).compute(run_target=target)
        result=part if result is None else result.merge(part,on=['date','code'],how='outer',validate='one_to_one')
    return result

def rank_universe(factors, market, config):
    dates=market['close'].index
    weekly=pd.Series(dates,index=dates).groupby(dates.to_period('W-FRI')).max()
    signals=[pd.Timestamp(d) for period,d in weekly.items()
        if (period.end_time.normalize()<=pd.Timestamp(config['signal_date']) or pd.Timestamp(d)==pd.Timestamp(config['signal_date'])) and d>=pd.Timestamp(config['analysis_start'])]
    adv=market['dollar'].rolling(config['adv_window'],min_periods=config['adv_window']).mean()
    parts=[]; skipped=[]
    for date in signals:
        frame=factors[factors.date==date].dropna(subset=['RSTR','DASTD','HBETA']).copy()
        frame['adv20_usd']=frame.code.map(adv.loc[date]); frame['last_volume']=frame.code.map(market['volume'].loc[date])
        frame=frame[(frame.adv20_usd>=config['adv_min_usd'])&(frame.last_volume>0)]
        frame=frame.merge(market['universe'],on='code',validate='one_to_one')
        frame=frame.sort_values(['adv20_usd','code'],ascending=[False,True]).drop_duplicates('issuer_id')
        if len(frame)<config['min_eligible']:
            if parts: raise ValueError(f'Universe unexpectedly fell below min_eligible on {date.date()}')
            skipped.append(str(date.date())); continue
        for name in ['RSTR','DASTD']:
            med=frame[name].median(); mad=(frame[name]-med).abs().median()
            clipped=frame[name].clip(med-3*1.4826*mad,med+3*1.4826*mad) if mad>0 else frame[name]
            sd=clipped.std(ddof=0)
            frame['z_'+name]=(clipped-clipped.mean())/sd if sd>0 else 0.
        parts.append(frame)
    if not parts: raise ValueError('No post-warmup signals')
    result=pd.concat(parts,ignore_index=True); result.attrs['warmup_skipped']=skipped
    if result.date.max()!=pd.Timestamp(config['signal_date']): raise ValueError('Latest signal is not a complete week-end signal')
    return result

def select_portfolio(rankings, config):
    parts=[]
    for date,source in rankings.groupby('date',sort=True):
        frame=source.copy()
        if config['score_method']=='zscore': mom,vol=frame.z_RSTR,-frame.z_DASTD
        else: mom,vol=frame.RSTR.rank(pct=True),frame.DASTD.rank(ascending=False,pct=True)
        frame['score']=config['weights']['RSTR']*mom+config['weights']['DASTD']*vol
        frame=frame.sort_values(['score','code'],ascending=[False,True])
        frame['rank']=range(1,len(frame)+1)
        if len(frame)<config['holdings']: raise ValueError('Insufficient eligible companies')
        boundary=frame.score.iloc[config['holdings']-1]
        frame['boundary_tie_count']=int((frame.score==boundary).sum())
        frame=frame.head(config['holdings']).copy()
        if not frame.issuer_id.is_unique: raise ValueError('Duplicate issuer')
        frame['target_weight']=1/config['holdings']
        parts.append(frame)
    if not parts: raise ValueError('No eligible selections')
    return pd.concat(parts,ignore_index=True)
