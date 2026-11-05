"""Frozen eight-way research experiment. All paths resolve from repository root."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_config():
    cfg=json.loads((HERE/"experiment_config.json").read_text())
    fixed={"recent_window":21,"medium_window":63,"old_weights":[.7,-.3],"new_weights":[.4,.3,-.3]}
    for key,value in fixed.items():
        if cfg.get(key)!=value:raise ValueError(f"This approved experiment requires {key}={value}")
    return cfg

def positive(values):
    a=np.asarray(values,dtype=float)
    if not np.isfinite(a).all() or (a<=0).any():raise ValueError("Missing, nonfinite or nonpositive price")

def read_pool(cfg):
    d=pd.read_csv(ROOT/cfg["rankings"])
    d=d[d.date==cfg["signal_date"]].copy()
    if d.empty or d.code.duplicated().any() or d.issuer_id.duplicated().any():raise ValueError("Invalid archived qualification pool")
    if d[["z_RSTR","z_DASTD","sector"]].isna().any().any():raise ValueError("Incomplete archived factors/sector")
    return d

def momenta(prices,signal):
    p=prices.loc[prices.index<=pd.Timestamp(signal)].sort_index()
    if p.index.duplicated().any() or len(p)<64 or p.index[-1]!=pd.Timestamp(signal):raise ValueError("Insufficient signal history")
    positive(p.iloc[-64:])
    return pd.DataFrame({"recent":p.iloc[-1]/p.iloc[-22]-1,"medium":p.iloc[-1]/p.iloc[-64]-1})

def relative_peers(returns,sectors):
    d=pd.DataFrame({"r":returns,"sector":sectors})
    if d.isna().any().any():raise ValueError("Missing peer inputs")
    counts=d.groupby("sector").r.transform("count")
    if (counts<2).any():raise ValueError("Industry has no other company")
    return d.r-(d.groupby("sector").r.transform("sum")-d.r)/(counts-1)

def standardize(s):
    med=s.median();mad=(s-med).abs().median()
    v=s.clip(med-3*1.4826*mad,med+3*1.4826*mad) if mad>0 else s
    sd=v.std(ddof=0)
    return (v-v.mean())/sd if sd>0 else v*0

def select(ranking,n,cap):
    d=ranking.sort_values(["score","code"],ascending=[False,True]).copy()
    d["rank"]=np.arange(1,len(d)+1)
    chosen=[];skipped=[];counts={};limit=math.floor(cap*n+1e-12) if cap is not None else n
    for _,row in d.iterrows():
        sector=row.sector
        if counts.get(sector,0)>=limit:
            skipped.append(row.to_dict());continue
        chosen.append(row.to_dict());counts[sector]=counts.get(sector,0)+1
        if len(chosen)==n:break
    if len(chosen)!=n:raise ValueError("Cannot fill target under industry limit")
    result=pd.DataFrame(chosen);result["target_weight"]=1/n
    return result,skipped

def check_baseline(s,cfg):
    old=pd.read_csv(ROOT/cfg["baseline"])
    if s.code.tolist()!=old.code.tolist():raise ValueError("G0 archived codes/order differ")
    np.testing.assert_allclose(s.score,old.score,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(s.target_weight,old.target_weight,rtol=0,atol=1e-12)

def freeze(run):
    cfg=load_config()
    if run.exists():raise FileExistsError(f"Run already exists: {run}")
    run.mkdir(parents=True)
    try:
        pool=read_pool(cfg);prices={};inputs=[ROOT/cfg["rankings"],ROOT/cfg["baseline"],HERE/"experiment_config.json",ROOT/cfg["snapshot"]/"universe.csv",ROOT/cfg["snapshot"]/"manifest.json"]
        calendar_path=ROOT/cfg["snapshot"]/"calendar_XNYS.csv"
        inputs.append(calendar_path)
        calendar=pd.read_csv(calendar_path)
        # Align every company to the same actual market dates, never per-stock rows.
        dates=pd.DatetimeIndex(pd.to_datetime(calendar.iloc[:,0]));dates=dates[dates<=pd.Timestamp(cfg["signal_date"])][-64:]
        for code in pool.code:
            path=ROOT/cfg["snapshot"]/"raw"/(code+".csv");inputs.append(path)
            d=pd.read_csv(path,parse_dates=["date"])
            if d.date.duplicated().any():raise ValueError(f"Duplicate dates: {code}")
            prices[code]=d.set_index("date")["Adj Close"].reindex(dates)
        m=momenta(pd.DataFrame(prices),cfg["signal_date"])
        m["relative"]=relative_peers(m.recent,pool.set_index("code").sector)
        m["z_recent"]=standardize(m.relative);m["z_medium"]=standardize(m.medium)
        pool=pool.join(m,on="code")
        pool["old_score"]=.7*pool.z_RSTR-.3*pool.z_DASTD
        pool["new_score"]=.4*pool.z_recent+.3*pool.z_medium-.3*pool.z_DASTD
        pool.to_csv(run/"scores.csv",index=False)
        selected=[];skips=[];ties={}
        for g in cfg["groups"]:
            ranking=pool.assign(score=pool[g["strategy"]+"_score"])
            s,k=select(ranking,g["n"],g["cap"])
            if g["id"]=="G0":check_baseline(s,cfg)
            s["group"]=g["id"];selected.append(s)
            skips.extend(dict(x,group=g["id"]) for x in k)
            ties[g["id"]]=ranking.loc[ranking.score.duplicated(False),["code","score"]].to_dict("records")
        pd.concat(selected).to_csv(run/"selected.csv",index=False)
        pd.DataFrame(skips).to_csv(run/"industry_skips.csv",index=False)
        dump(run/"config.json",cfg)
        dump(run/"freeze_manifest.json",{"created_utc":datetime.now(timezone.utc).isoformat(),"pool_size":len(pool),"inputs":{str(p.relative_to(ROOT)):sha(p) for p in inputs},"artifacts":{n:sha(run/n) for n in ["scores.csv","selected.csv","industry_skips.csv","config.json"]},"ties":ties,"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__})
    except Exception as e:
        dump(run/"freeze_failure.json",{"error":str(e)});raise
    print(run)

def verify_freeze(run):
    m=json.loads((run/"freeze_manifest.json").read_text())
    for name,digest in m["artifacts"].items():
        if sha(run/name)!=digest:raise ValueError(f"Frozen artifact changed: {name}")
    return json.loads((run/"config.json").read_text())

def download(run):
    cfg=verify_freeze(run)
    out=run/"market"
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    import yfinance as yf
    yf.config.debug.hide_exceptions=False
    codes=sorted(set(pd.read_csv(run/"selected.csv").code)|{"SPY"})
    manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"yfinance":yf.__version__,"start":cfg["download_start"],"end_exclusive":cfg["download_end_exclusive"],"auto_adjust":False,"codes":codes,"freeze_manifest_sha256":sha(run/"freeze_manifest.json"),"files":{}}
    try:
        for code in codes:
            d=yf.Ticker(code).history(start=cfg["download_start"],end=cfg["download_end_exclusive"],auto_adjust=False,actions=True)
            if d.empty:raise ValueError(f"No market data: {code}")
            d.index=pd.to_datetime(d.index.date);d.index.name="date"
            d.to_csv(out/(code+".csv"));manifest["files"][code+".csv"]=sha(out/(code+".csv"))
        load_market(run,cfg,codes)
        dump(run/"market_quality.json",{"status":"complete","codes":len(codes),"required_dates":cfg["valuation_dates"],"price_basis":"Open * Adj Close / Close; Adj Close valuation"})
        dump(run/"download_manifest.json",manifest)
    except Exception as e:
        dump(run/"download_failure.json",dict(manifest,error=str(e)));raise
    print(out)

def load_market(run,cfg,codes):
    close={};entry={};dates=pd.to_datetime(cfg["valuation_dates"])
    for code in codes:
        d=pd.read_csv(run/"market"/(code+".csv"),parse_dates=["date"])
        if d.date.duplicated().any():raise ValueError(f"Duplicate market dates: {code}")
        d=d.set_index("date").reindex(dates)
        positive(d[["Open","Close","Adj Close"]])
        close[code]=d["Adj Close"]
        e=d.loc[pd.Timestamp(cfg["entry_date"])];entry[code]=e.Open*e["Adj Close"]/e.Close
    return pd.DataFrame(close),pd.Series(entry)

def portfolio_nav(close,entry):
    positive(close);positive(entry)
    if set(close.columns)!=set(entry.index):raise ValueError("Entry universe mismatch")
    return close.div(entry,axis=1).mean(axis=1)

def max_drawdown(nav):
    v=np.r_[1.,np.asarray(nav,dtype=float)]
    return float(np.min(v/np.maximum.accumulate(v)-1))


def evaluate(run,publish=False):
    cfg=verify_freeze(run)
    if (run/"metrics.json").exists():raise FileExistsError("Evaluation already exists")
    names=["report.md","metrics.json","selected.csv","comparison.png"]
    if publish and any((HERE/n).exists() for n in names):raise FileExistsError("Published artifacts already exist")
    dm=json.loads((run/"download_manifest.json").read_text())
    if dm["freeze_manifest_sha256"]!=sha(run/"freeze_manifest.json"):raise ValueError("Download freeze mismatch")
    for name,digest in dm["files"].items():
        if sha(run/"market"/name)!=digest:raise ValueError("Market hash mismatch: "+name)
    selected=pd.read_csv(run/"selected.csv")
    codes=sorted(set(selected.code)|{"SPY"})
    if codes!=dm["codes"]:raise ValueError("Download universe mismatch")
    close,entry=load_market(run,cfg,codes)
    metrics={};navs={};details=[];sectors=[]
    for name in [g["id"] for g in cfg["groups"]]+["SPY"]:
        s=selected[selected.group==name] if name!="SPY" else pd.DataFrame({"code":["SPY"],"sector":["SPY"]})
        tickers=s.code.tolist();n=len(tickers)
        gross=portfolio_nav(close[tickers],entry[tickers]);net=gross/(1+cfg["cost"])
        navs[name+" gross"]=gross;navs[name+" net"]=net
        returns=close[tickers].iloc[-1]/entry[tickers]-1
        contribution=returns/n
        if not np.isclose(contribution.sum(),gross.iloc[-1]-1):raise ValueError("Contribution mismatch")
        metrics[name]={"gross_return":float(gross.iloc[-1]-1),"net_return":float(net.iloc[-1]-1),"max_close_drawdown_gross":max_drawdown(gross),"max_close_drawdown_net":max_drawdown(net),"up_stocks":int((returns>0).sum()),"stocks":n}
        for _,r in s.iterrows():
            details.append({"group":name,"code":r.code,"sector":r.sector,"entry_adjusted_open":float(entry[r.code]),"final_adjusted_close":float(close[r.code].iloc[-1]),"return":float(returns[r.code]),"gross_contribution":float(contribution[r.code]),"score":None if name=="SPY" else float(r.score),"rank":None if name=="SPY" else int(r["rank"])})
        for sector,ss in s.groupby("sector"):
            sectors.append({"group":name,"sector":sector,"target_weight":len(ss)/n,"gross_contribution":float(contribution[ss.code].sum())})
    for m in metrics.values():
        m["excess_spy_gross"]=m["gross_return"]-metrics["SPY"]["gross_return"]
        m["improvement_g0_gross"]=m["gross_return"]-metrics["G0"]["gross_return"]
    pd.DataFrame(navs).to_csv(run/"daily_nav.csv",index_label="date")
    pd.DataFrame(details).to_csv(run/"stock_contributions.csv",index=False)
    pd.DataFrame(sectors).to_csv(run/"sector_contributions.csv",index=False)
    dump(run/"metrics.json",metrics)
    import os
    cache=run/"cache"
    mpl_cache=cache/"matplotlib";xdg_cache=cache/"xdg"
    mpl_cache.mkdir(parents=True,exist_ok=True);xdg_cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR",str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME",str(xdg_cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    labels=list(metrics);axes[0].bar(labels,[100*metrics[k]["gross_return"] for k in labels],label="Gross",alpha=.7)
    axes[0].scatter(labels,[100*metrics[k]["net_return"] for k in labels],color="black",s=16,label="Net")
    axes[0].axhline(0,color="grey",lw=.7);axes[0].set_ylabel("Return (%)");axes[0].legend()
    for k in labels:
        axes[1].plot(["Initial"]+cfg["valuation_dates"],[1]+navs[k+" net"].tolist(),marker="o",label=k)
    axes[1].tick_params(axis="x",rotation=25);axes[1].set_ylabel("Net NAV");axes[1].legend(ncol=3)
    fig.suptitle("Eight strategies: Nov 2 open to Nov 4 close, 2026")
    fig.tight_layout();fig.savefig(run/"comparison.png",dpi=160);plt.close(fig)
    pct=lambda x:f"{100*x:.3f}%"
    rows=["# 10月30日选股：8组对照研究","", "信号日2026-10-30，11月2日调整开盘买入，持有至11月4日调整收盘。各组买入后持有，股票内部等权，使用分数股。", "", "这是看过行情后的三天诊断，不能确定三个月比赛的最优策略；未自动修改正式策略。", "", "|组别|评分|只数|行业上限|毛收益|净收益|相对SPY毛收益差|比G0改善|净值收盘最大回撤|上涨数|", "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for g in cfg["groups"]:
        m=metrics[g["id"]];rows.append(f"|{g['id']}|{'原策略' if g['strategy']=='old' else '新策略'}|{g['n']}|{'30%' if g['cap'] else '无'}|{pct(m['gross_return'])}|{pct(m['net_return'])}|{100*m['excess_spy_gross']:+.3f}个百分点|{100*m['improvement_g0_gross']:+.3f}个百分点|{pct(m['max_close_drawdown_net'])}|{m['up_stocks']}|")
    best=max((g["id"] for g in cfg["groups"]),key=lambda k:metrics[k]["gross_return"])
    best_delta=100*metrics[best]["excess_spy_gross"]
    rows += ["",f"SPY毛收益{pct(metrics['SPY']['gross_return'])}，同口径净收益{pct(metrics['SPY']['net_return'])}。", "",f"本次毛收益最高的是{best}（{pct(metrics[best]['gross_return'])}），相对SPY{'领先' if best_delta>0 else '落后' if best_delta<0 else '持平'}{abs(best_delta):.3f}个百分点。这只说明这三天的结果，不能推断以后表现。", "", "## 三项变化的配对比较", ""]
    for title,pairs in [("增加至15只",[(0,1),(2,3),(4,5),(6,7)]),("改用新评分",[(0,2),(1,3),(4,6),(5,7)]),("增加30%行业上限",[(0,4),(1,5),(2,6),(3,7)])]:
        rows.append(title+"：")
        for a,b in pairs:
            delta=100*(metrics[f"G{b}"]["gross_return"]-metrics[f"G{a}"]["gross_return"])
            rows.append(f"- G{b}相对G{a}：{'改善' if delta>0 else '收益降低' if delta<0 else '无变化，差值'}{abs(delta):.3f}个百分点。")
        rows.append("")
    for g in cfg["groups"]:
        s=selected[selected.group==g["id"]]
        rows += [f"## {g['id']}名单与行业分配", "", "、".join(s.code), "", "；".join(f"{r['sector']} {pct(r['target_weight'])}，收益贡献{100*r['gross_contribution']:+.3f}个百分点" for r in sectors if r['group']==g['id']), ""]
    rows += ["## 口径与限制", "", "原策略为70%原动量减30%波动标准分。新策略为40%相对同行的1个月动量、30%3个月绝对动量、减30%原波动标准分；同行为原资格池同GICS行业的其他公司。两个动量窗口重叠。行业上限只约束期初目标，不触发再平衡。", "", "净收益统一假设从现金新建仓，买入费10bps，投入本金1/1.001，期末不卖出。收盘最大回撤包含初始本金1，仅代表三次收盘观察，不包含盘中回撤；不年化。逐股收益及组合贡献见运行目录stock_contributions.csv。", "", "资格池沿用11月1日供应商快照和原资格过滤，公司名单不是严格逐日历史名单。选股输入截断于10月30日；估值行情在冻结名单后统一重新下载，供应商可能修订历史复权价格。G0与原名单、顺序、权重和评分已核对，约-6.15%毛收益仅作为先前复盘参考，差异不强行校准。", "", f"完整审计与来源：`{run.relative_to(ROOT)}/freeze_manifest.json`、`download_manifest.json`、`evaluation_manifest.json`。", "", "![8组收益和净值](comparison.png)", ""]
    (run/"report.md").write_text("\n".join(rows))
    dump(run/"evaluation_manifest.json",{"created_utc":datetime.now(timezone.utc).isoformat(),"freeze_manifest_sha256":sha(run/"freeze_manifest.json"),"download_manifest_sha256":sha(run/"download_manifest.json"),"artifacts":{n:sha(run/n) for n in names+["daily_nav.csv","stock_contributions.csv","sector_contributions.csv"]}})
    if publish:
        import shutil
        for n in names:shutil.copyfile(run/n,HERE/n)
    print(pd.DataFrame(metrics).T.to_string());print(run/"report.md")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["freeze","download","evaluate"])
    parser.add_argument("--run-dir",required=True,type=Path)
    parser.add_argument("--publish",action="store_true")
    args=parser.parse_args();run=args.run_dir.resolve()
    out=(ROOT/load_config()["output_root"]).resolve()
    if run.parent!=out:raise ValueError("Run must be a direct child of configured output root")
    if args.stage=="freeze":freeze(run)
    elif args.stage=="download":download(run)
    else:evaluate(run,args.publish)

if __name__=="__main__":main()
