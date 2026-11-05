"""Offline regression checks; no downloads."""
import unittest
import numpy as np
import pandas as pd
import compare_strategies as c

class ExperimentTests(unittest.TestCase):
    def test_windows_future_and_peers(self):
        dates=pd.bdate_range(end="2026-10-30",periods=64)
        p=pd.DataFrame({"A":np.arange(1,65),"B":np.ones(64)*2},index=dates)
        r=c.momenta(p,"2026-10-30")
        self.assertAlmostEqual(r.loc["A","medium"],63)
        self.assertAlmostEqual(r.loc["A","recent"],64/43-1)
        extended=pd.concat([p,p.iloc[[-1]].set_axis([pd.Timestamp("2026-11-02")])*100])
        pd.testing.assert_frame_equal(r,c.momenta(extended,"2026-10-30"))
        rel=c.relative_peers(r.recent,pd.Series({"A":"x","B":"x"}))
        self.assertAlmostEqual(rel.A,r.loc["A","recent"])
        self.assertAlmostEqual(rel.B,-r.loc["A","recent"])
        with self.assertRaises(ValueError):c.relative_peers(r.recent,pd.Series({"A":"x","B":"y"}))
    def test_caps(self):
        d=pd.DataFrame({"code":[f"S{i:02}" for i in range(30)],"sector":[str(i//5) for i in range(30)],"score":range(30,0,-1)})
        for n,cap in [(10,3),(15,4)]:
            s,skip=c.select(d,n,.3)
            self.assertEqual(len(s),n);self.assertLessEqual(s.sector.value_counts().max(),cap);self.assertTrue(skip)
        with self.assertRaises(ValueError):c.select(d.iloc[:5],10,.3)
    def test_cost_drawdown_missing(self):
        nav=c.portfolio_nav(pd.DataFrame({"A":[90.,100.]}),pd.Series({"A":100.}))
        self.assertAlmostEqual(c.max_drawdown(nav),-.1)
        flat=c.portfolio_nav(pd.DataFrame({"A":[100.]}),pd.Series({"A":100.}))/1.001
        self.assertAlmostEqual(flat.iloc[-1]-1,1/1.001-1)
        self.assertAlmostEqual(flat.iloc[-1]+.001/1.001,1)
        with self.assertRaises(ValueError):c.portfolio_nav(pd.DataFrame({"A":[np.nan]}),pd.Series({"A":100.}))
    def test_standardization(self):
        flat=pd.Series([2.,2.,2.])
        np.testing.assert_allclose(c.standardize(flat),0)
        raw=pd.Series([1.,2.,3.,4.,100.])
        expected=raw.clip(3-3*1.4826,3+3*1.4826)
        np.testing.assert_allclose(c.standardize(raw),(expected-expected.mean())/expected.std(ddof=0))
    def test_bad_history(self):
        dates=pd.bdate_range(end="2026-10-30",periods=64)
        prices=pd.DataFrame({"A":np.ones(64)},index=dates)
        for bad in [prices.iloc[1:],pd.concat([prices,prices.iloc[[-1]]]),prices.assign(A=0)]:
            with self.assertRaises(ValueError):c.momenta(bad,"2026-10-30")
    def test_market_basis_and_missing_day(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);(run/"market").mkdir()
            dates=["2026-11-02","2026-11-03","2026-11-04"]
            data=pd.DataFrame({"date":dates,"Open":[100,100,100],"Close":[100,100,100],"Adj Close":[50,55,60]})
            data.to_csv(run/"market/A.csv",index=False)
            cfg={"entry_date":dates[0],"valuation_dates":dates}
            close,entry=c.load_market(run,cfg,["A"])
            self.assertEqual(entry.A,50)
            self.assertAlmostEqual(c.portfolio_nav(close,entry).iloc[-1]-1,.2)
            data.iloc[:2].to_csv(run/"market/A.csv",index=False)
            with self.assertRaises(ValueError):c.load_market(run,cfg,["A"])
    def test_g0_archive(self):
        cfg=c.load_config(); ranks=c.read_pool(cfg); ranks["score"]=.7*ranks.z_RSTR-.3*ranks.z_DASTD
        s,_=c.select(ranks,10,None);c.check_baseline(s,cfg)
        s15,_=c.select(ranks,15,None)
        self.assertEqual(s.code.tolist(),s15.code.iloc[:10].tolist())
        np.testing.assert_allclose(s.score,s15.score.iloc[:10])

if __name__=="__main__":unittest.main(verbosity=2)
