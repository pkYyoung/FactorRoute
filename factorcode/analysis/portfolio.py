"""日频研究组合：调整价、分数股、全额成交的近似模拟。"""
import numpy as np
import pandas as pd
from . import metrics as m

def simulate(close, op, schedule, cost):
    symbols=close.columns
    units=pd.Series(0.,index=symbols); cash=1.; records=[]; trades=[]
    if not schedule:
        return pd.Series(1., index=close.index, name='nav'), pd.DataFrame(columns=['date','nav_before','traded_value','cost','cost_rate'])
    if not close.index.equals(op.index) or not close.columns.equals(op.columns):
        raise ValueError('Opening and closing panels must align')
    if not np.isfinite(cost) or not 0 <= cost < .1: raise ValueError('Invalid cost')
    first=min(schedule)
    for date in close.index[close.index>=first]:
        if date in schedule:
            current=units*op.loc[date]
            # 非持仓缺报价不影响账户；持仓缺报价则停止，不能填零掩盖损失。
            if current[units!=0].isna().any(): raise ValueError('持仓开盘价格缺失')
            current=current.fillna(0.)
            capital=float(current.sum()+cash)
            weights=schedule[date].reindex(symbols,fill_value=0.)
            if not np.isclose(weights.sum(),1.) or not (weights>=0).all():
                raise ValueError('Weights must be nonnegative and sum to one')
            if op.loc[date,weights>0].isna().any(): raise ValueError('目标股开盘缺失')
            # 解投入金额+按实际差额收取的成本=调仓前净值，保证资金自洽。
            invest=capital
            for _ in range(30):
                new=capital-cost*float((weights*invest-current).abs().sum())
                if abs(new-invest)<1e-14: break
                invest=new
            traded=float((weights*invest-current).abs().sum()); fee=cost*traded
            if abs(invest+fee-capital)>=1e-11: raise RuntimeError('Budget conservation failed')
            units=(weights*invest/op.loc[date]).fillna(0.)
            cash=capital-invest-fee
            if cash < -1e-11: raise RuntimeError('Negative cash after fees')
            trades.append({'date':date,'nav_before':capital,'traded_value':traded,'cost':fee,'cost_rate':cost})
        marked=units*close.loc[date]
        if marked[units!=0].isna().any(): raise ValueError('持仓收盘价格缺失')
        records.append({'date':date,'nav':float(marked.sum()+cash)})
    return pd.DataFrame(records).set_index('date').nav, pd.DataFrame(trades)

def stats(nav):
    if len(nav)==0: raise ValueError('Empty NAV')
    ret=nav.pct_change(fill_method=None).iloc[1:]
    total=m.cumulative_return(ret); annual=m.annualized_return(total,len(ret))
    vol=m.annualized_volatility(ret)
    # 原max_drawdown函数只从第一笔收益后的净值开始找峰值；补0收益纳入初始本金1。
    dd=m.max_drawdown(pd.concat([pd.Series([0.]),ret.reset_index(drop=True)],ignore_index=True))
    draw=nav/nav.cummax()-1; trough=draw.idxmin(); peak=nav.loc[:trough].idxmax()
    recovered=nav.loc[trough:][nav.loc[trough:]>=nav.loc[peak]]
    return {'final_nav':float(nav.iloc[-1]),'total_return':total if len(ret) else 0.,'annualized_return_252':annual,
      'annualized_volatility':vol,'sharpe_rf0':float(ret.mean()/ret.std(ddof=1)*np.sqrt(252)) if ret.std(ddof=1)>0 else None,
      'max_drawdown':dd,'calmar':m.calmar_ratio(annual,dd),'daily_win_rate':m.win_rate(ret),
      'trading_days':len(ret),'drawdown_peak':str(peak.date()),'drawdown_trough':str(trough.date()),
      'drawdown_recovered':None if recovered.empty else str(recovered.index[0].date())}

def build_schedule(picks, dates):
    result={}
    for signal, group in picks.groupby('date'):
        future=dates[dates>signal]
        if len(future): result[future[0]]=group.set_index('code').target_weight
    return result

def benchmark_nav(benchmark, first, baseline, last, cost):
    opening=benchmark.Open*benchmark['Adj Close']/benchmark.Close
    nav=benchmark.loc[first:last,'Adj Close']/opening.loc[first]/(1+cost)
    return pd.concat([pd.Series([1.],index=[baseline]),nav])

def plot_nav(navs, out, title):
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    fig,(ax,bx)=plt.subplots(2,1,figsize=(12,8),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[2,1]})
    colors={'strategy':'#2463eb','before_cost':'#8eb1fa','SPY':'#717b89'}
    for name,series in navs.items():
        ax.plot(series.index,series,label=name,color=colors.get(name),ls='--' if name=='before_cost' else '-')
        if name!='before_cost': bx.plot(series.index,series/series.cummax()-1,color=colors.get(name))
    ax.set_title(title,loc='left',fontsize=12); ax.set_ylabel('Initial 1 unit'); ax.legend()
    bx.set_ylabel('Drawdown'); bx.yaxis.set_major_formatter(PercentFormatter(1))
    bx.set_xlabel('US market close | Current-constituent research approximation')
    for item in (ax,bx): item.grid(alpha=.2)
    path=Path(out)/'equity.png'; fig.savefig(path,dpi=160); plt.close(fig)
    return path
