"""Yahoo日线适配。保留供应商原始快照；调整研究价格不作执行报价。"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import io
import json
import time
import pandas as pd

REQUIRED=['Open','High','Low','Close','Adj Close','Volume','Dividends','Stock Splits']

def request_batch(symbols, config):
    try: import yfinance as yf
    except ImportError as exc:
        raise RuntimeError('Install factorroute[yfinance] for download stage') from exc
    return yf.download(symbols,start=config['download_start'],end=config['download_end_exclusive'],
        interval='1d',actions=True,auto_adjust=False,repair=False,keepna=True,threads=2,
        progress=False,timeout=15,group_by='ticker')

def download_snapshot(config, dest):
    try:
        import yfinance as yf
        import requests
    except ImportError as exc:
        raise RuntimeError('Install factorroute[yfinance] for download stage') from exc
    dest=Path(dest); dest.mkdir(parents=True,exist_ok=False); (dest/'raw').mkdir()
    yf.set_tz_cache_location(str(dest/'session_cache'))
    if config.get('universe_file'):
        path=config['_root']/config['universe_file']; universe=pd.read_csv(path,dtype=str)
        source=config['universe_file']
    else:
        urls=[config['universe_url'],'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv']
        errors=[]
        for source in urls:
            try:
                response=requests.get(source,timeout=25,headers={'User-Agent':'FactorRoute research/0.1'})
                response.raise_for_status()
                if source.endswith('.csv'):
                    universe=pd.read_csv(io.StringIO(response.text),dtype=str).rename(columns={'Name':'Security','Sector':'GICS Sector'})
                    (dest/'raw/universe_source.csv').write_text(response.text)
                else:
                    universe=pd.read_html(io.StringIO(response.text),attrs={'id':'constituents'})[0]
                    (dest/'raw/universe_source.html').write_text(response.text)
                if not 450<=len(universe)<=550: raise ValueError('Unexpected S&P500 size')
                break
            except Exception as exc: errors.append(f'{type(exc).__name__}: {exc}')
        else: raise RuntimeError(f'Universe retrieval failed: {errors}')
    if not {'Symbol','Security','GICS Sector','CIK'}.issubset(universe): raise ValueError('Universe fields missing')
    universe['yahoo_symbol']=universe.Symbol.str.replace('.','-',regex=False)
    universe['issuer_id']=universe.CIK.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
    if not universe.yahoo_symbol.is_unique: raise ValueError('Duplicate universe codes')
    universe.to_csv(dest/'universe.csv',index=False)
    symbols=universe.yahoo_symbol.tolist()+[config.get('benchmark','SPY')]
    log=[]
    def fetch(group):
        tick=time.monotonic()
        try:
            frame=request_batch(group,config)
            if frame is not None and not frame.empty:
                for symbol in group:
                    if isinstance(frame.columns,pd.MultiIndex):
                        if symbol not in frame.columns.get_level_values(0): continue
                        one=frame[symbol].dropna(how='all')
                    else: one=frame.dropna(how='all')
                    if not one.empty: one.to_csv(dest/'raw'/f'{symbol}.csv',index_label='date')
            log.append({'symbols':group,'seconds':round(time.monotonic()-tick,3)})
        except Exception as exc:
            log.append({'symbols':group,'error':str(exc),'seconds':round(time.monotonic()-tick,3)})
    for n in range(0,len(symbols),20):
        fetch(symbols[n:n+20]); time.sleep(1)
    missing=[s for s in symbols if not (dest/'raw'/f'{s}.csv').exists()]
    if len(missing)<=15:
        for symbol in missing: fetch([symbol]); time.sleep(1)
    missing=[s for s in symbols if not (dest/'raw'/f'{s}.csv').exists()]
    manifest={'retrieved_utc':datetime.now(timezone.utc).isoformat(),'universe_source':source,
        'start':config['download_start'],'end_exclusive':config['download_end_exclusive'],
        'historical_membership':False,'yfinance':yf.__version__,'pandas':pd.__version__,
        'missing_after_retry':missing,'requests':log,
        'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (dest/'raw').glob('*.csv')}}
    (dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    if missing: raise RuntimeError(f'Download incomplete: {missing}; see saved manifest')
    load_snapshot(dest,config['signal_date'])
    return dest

def _read(path, signal_date):
    frame=pd.read_csv(path,index_col=0,parse_dates=True)
    frame.index=pd.DatetimeIndex(frame.index).tz_localize(None)
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f'Invalid date index: {Path(path).name}')
    return frame.loc[:signal_date]

def load_snapshot(path, signal_date):
    path=Path(path)
    if not (path/'universe.csv').exists():
        raise FileNotFoundError('Snapshot unavailable: run --stage download or choose a local snapshot')
    universe=pd.read_csv(path/'universe.csv',dtype={'issuer_id':str,'CIK':str})
    universe['code']=universe.yahoo_symbol if 'yahoo_symbol' in universe else universe.Symbol.str.replace('.','-',regex=False)
    universe['company']=universe.Security; universe['sector']=universe['GICS Sector']
    ids=universe['issuer_id'] if 'issuer_id' in universe else universe.CIK
    universe['issuer_id']=ids.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(10)
    universe=universe[['code','company','sector','issuer_id']]
    if not universe.code.is_unique: raise ValueError('Duplicate universe codes')
    benchmark=_read(path/'raw/SPY.csv',signal_date)
    if benchmark.empty or benchmark.index.max()!=pd.Timestamp(signal_date): raise ValueError('SPY signal date is stale or unavailable')
    if not set(REQUIRED).issubset(benchmark) or benchmark[REQUIRED].isna().any().any(): raise ValueError('Invalid benchmark fields')
    dates=benchmark.index
    calendar_status='not independently verified: install download extra'
    try:
        import exchange_calendars as xc
        expected=xc.get_calendar('XNYS').sessions_in_range(dates.min(),pd.Timestamp(signal_date)).tz_localize(None)
        if not dates.equals(expected): raise ValueError('Benchmark dates disagree with XNYS calendar')
        week=pd.Timestamp(signal_date).to_period('W-FRI')
        week_sessions=xc.get_calendar('XNYS').sessions_in_range(week.start_time.normalize(),week.end_time.normalize()).tz_localize(None)
        if week_sessions.empty or week_sessions.max()!=pd.Timestamp(signal_date):
            raise ValueError('signal_date must be the final market session of its week')
        calendar_status='XNYS verified'
    except ImportError:
        if pd.Timestamp(signal_date).weekday()!=4:
            raise ValueError('Install factorroute[yfinance] to verify a non-Friday week-end signal')
    panels={k:{} for k in ['Adj Close','Open','Close','Volume']}; quality=[]
    for symbol in universe.code:
        q={'code':symbol,'status':'excluded'}
        try:
            d=_read(path/'raw'/f'{symbol}.csv',signal_date)
            if d.empty or not set(REQUIRED).issubset(d): raise ValueError('Empty data or missing fields')
            price=d[['Open','High','Low','Close']]
            q.update(rows=len(d),first=str(d.index.min().date()),last=str(d.index.max().date()),
                duplicates=0,missing_ohlcv=int(d[['Open','High','Low','Close','Adj Close','Volume']].isna().any(axis=1).sum()),
                bad_ohlc=int(((d.High<price.max(axis=1))|(d.Low>price.min(axis=1))|(price<=0).any(axis=1)).sum()),
                negative_volume=int((d.Volume<0).sum()),zero_volume=int((d.Volume==0).sum()),
                internal_missing_sessions=len(dates[(dates>=d.index.min())&(dates<=d.index.max())].difference(d.index)))
            if q['bad_ohlc'] or q['negative_volume'] or q['missing_ohlcv'] or q['internal_missing_sessions']:
                raise ValueError('Price/volume or internal-date anomaly')
            if (d['Adj Close']<=0).any() or d.index.max()!=pd.Timestamp(signal_date): raise ValueError('Invalid adjusted close or stale quote')
            if len(d.index.difference(dates)): raise ValueError('Unexpected market dates')
            for key in panels: panels[key][symbol]=d[key].reindex(dates)
            q['status']='usable'
        except (OSError,ValueError,KeyError) as exc: q['reason']=str(exc)
        quality.append(q)
    if len(panels['Adj Close'])<.95*len(universe): raise ValueError('Usable universe coverage below 95%; inspect snapshot')
    close=pd.DataFrame(panels['Adj Close'],index=dates)
    opening=pd.DataFrame(panels['Open'],index=dates)*close/pd.DataFrame(panels['Close'],index=dates)
    return {'close':close,'open':opening,'volume':pd.DataFrame(panels['Volume'],index=dates),
        'dollar':pd.DataFrame(panels['Close'],index=dates)*pd.DataFrame(panels['Volume'],index=dates),
        'benchmark':benchmark,'universe':universe,'quality':pd.DataFrame(quality),'calendar_status':calendar_status}
