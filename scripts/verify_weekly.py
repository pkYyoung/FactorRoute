"""无网络合成验证：配置、评分、账户资金、日期及公开输出边界。"""
from pathlib import Path
import sys, json, tempfile, unittest
import subprocess,os
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from factorcode.weekly_config import load_config
import numpy as np
import pandas as pd
from factorcode.analysis.portfolio import simulate, stats, build_schedule
from factorcode.strategy.weekly import select_portfolio, rank_universe
from factorcode.weekly_workflow import publish_week

class ConfigTests(unittest.TestCase):
    def test_invalid_weight_fails_before_reading_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'week.json'
            path.write_text(json.dumps({'weights': {'RSTR': .7, 'DASTD': .4}}))
            with self.assertRaisesRegex(ValueError, 'weights'):
                load_config(path)

class PortfolioTests(unittest.TestCase):
    def test_flat_prices_charge_only_actual_entry(self):
        dates=pd.bdate_range('2024-01-01',periods=4)
        prices=pd.DataFrame({'A':100.,'B':100.},index=dates)
        w=pd.Series({'A':.5,'B':.5})
        nav,trades=simulate(prices,prices,{dates[0]:w,dates[2]:w},.001)
        self.assertTrue(np.allclose(nav,1/1.001))
        self.assertAlmostEqual(trades.cost.iloc[1],0.)

    def test_last_friday_has_no_fabricated_monday(self):
        dates=pd.to_datetime(['2026-10-29','2026-10-30'])
        picks=pd.DataFrame({'date':[dates[-1]],'code':['A'],'target_weight':[1.]})
        self.assertEqual(build_schedule(picks,dates),{})

    def test_drawdown_includes_initial_capital(self):
        nav=pd.Series([1.,.8,1.],index=pd.bdate_range('2024-01-01',periods=3))
        self.assertAlmostEqual(stats(nav)['max_drawdown'],-.2)

    def test_missing_held_quote_stops(self):
        dates=pd.bdate_range('2024-01-01',periods=3)
        prices=pd.DataFrame({'A':[100.,np.nan,100.]},index=dates)
        with self.assertRaises(ValueError):
            simulate(prices,prices,{dates[0]:pd.Series({'A':1.})},0.)

    def test_prefix_and_weight_drift(self):
        dates=pd.bdate_range('2024-01-01',periods=5)
        prices=pd.DataFrame({'A':[100.,110.,120.,115.,125.],'B':[100.,90.,80.,85.,75.]},index=dates)
        weights=pd.Series({'A':.5,'B':.5})
        schedule={dates[0]:weights,dates[3]:weights}
        full,_=simulate(prices,prices,schedule,.001)
        prefix,_=simulate(prices.iloc[:3],prices.iloc[:3],{dates[0]:weights},.001)
        self.assertTrue(np.allclose(prefix,full.iloc[:3]))
        self.assertGreater(full.iloc[-1],0)

class ScoringTests(unittest.TestCase):
    def test_weights_change_names_and_new_rank(self):
        frame=pd.DataFrame({'date':pd.to_datetime(['2026-10-30']*3),'code':['A','B','C'],
            'issuer_id':['1','2','3'],'z_RSTR':[2.,1.,0.],'z_DASTD':[2.,0.,-1.],
            'rank':[99,98,97],'target_weight':[.05,.05,.05]})
        config={'weights':{'RSTR':1.,'DASTD':0.},'score_method':'zscore','holdings':1}
        a=select_portfolio(frame,config)
        config['weights']={'RSTR':0.,'DASTD':1.}
        b=select_portfolio(frame,config)
        self.assertEqual(a.code.tolist(),['A']); self.assertEqual(b.code.tolist(),['C'])
        self.assertEqual(a['rank'].tolist(),[1]); self.assertEqual(a.target_weight.tolist(),[1.])

    def test_company_duplicate_removed_by_current_liquidity(self):
        dates=pd.bdate_range('2026-10-26',periods=5)
        columns=['A','A2','B']
        market={'close':pd.DataFrame(100.,index=dates,columns=columns),
            'volume':pd.DataFrame(100.,index=dates,columns=columns),
            'dollar':pd.DataFrame({'A':[200.]*5,'A2':[100.]*5,'B':[150.]*5},index=dates),
            'universe':pd.DataFrame({'code':columns,'issuer_id':['1','1','2'],'company':columns,'sector':['S']*3})}
        factors=pd.DataFrame({'date':[dates[-1]]*3,'code':columns,'RSTR':[2.,2.,1.],
            'DASTD':[1.,1.,1.],'HBETA':[1.,1.,1.]})
        config={'adv_window':2,'adv_min_usd':1,'min_eligible':2,'analysis_start':'2026-10-01','signal_date':'2026-10-30'}
        ranks=rank_universe(factors,market,config)
        self.assertEqual(set(ranks.code),{'A','B'})

class WorkflowTests(unittest.TestCase):
    def test_cli_help_from_other_directory(self):
        script=Path(__file__).resolve().with_name('run_weekly.py')
        with tempfile.TemporaryDirectory() as folder:
            result=subprocess.run([sys.executable,str(script),'--help'],cwd=folder,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--stage',result.stdout)

    def test_missing_snapshot_is_clear_and_offline(self):
        from factorcode.data.yfinance_source import load_snapshot
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError,'Snapshot unavailable'):
                load_snapshot(Path(folder),'2026-10-30')

    def test_public_export_drops_account_and_legacy_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'source'; target=Path(folder)/'public'; source.mkdir()
            pd.DataFrame({'date':['2026-10-30'],'code':['A'],'company':['Company'],
                'sector':['Sector'],'score':[1.],'rank':[1],'target_weight':[1.],
                'reference_shares':[123],'reference_budget_hkd':[99999],
                'private_path':['private-local-path']}).to_csv(source/'selected_internal.csv',index=False)
            summary={'signal_date':'2026-10-30','valuation_end':'2026-10-30','metrics':{},
                'rebalance_count':0,'execution_status':'pending','calendar_check':'synthetic',
                'sector_weights':{},'mean_beta_proxy':1,'boundary_tie_count':1,'software':{},
                'limitations':[],'private_budget':99999,'parameters':{'weights':{},'directions':{},
                    'score_method':'zscore','holdings':1,'cost_rate':.001}}
            (source/'summary.json').write_text(json.dumps(summary)); (source/'equity.png').write_bytes(b'synthetic-plot')
            publish_week(source,target)
            self.assertEqual(pd.read_csv(target/'selected.csv').columns.tolist(),
                ['date','code','company','sector','score','rank','target_weight'])
            self.assertNotIn('private_budget',(target/'metrics.json').read_text())
            with self.assertRaises(FileExistsError): publish_week(source,target)

class CalendarTests(unittest.TestCase):
    def setUp(self):
        import importlib.util
        if importlib.util.find_spec('exchange_calendars') is None:
            self.skipTest('Optional calendar package unavailable in base-only environment')

    def fixture(self,folder):
        root=Path(folder); (root/'raw').mkdir()
        pd.DataFrame({'Symbol':['A','B'],'Security':['A','B'],'GICS Sector':['S','S'],
            'CIK':['1','2']}).to_csv(root/'universe.csv',index=False)
        dates=pd.to_datetime(['2026-03-30','2026-03-31','2026-04-01','2026-04-02'])
        frame=pd.DataFrame({k:100. for k in ['Open','High','Low','Close','Adj Close','Volume']},index=dates)
        frame['Dividends']=0.; frame['Stock Splits']=0.
        for symbol in ['A','B','SPY']: frame.to_csv(root/'raw'/f'{symbol}.csv')
        return root

    def test_holiday_friday_uses_thursday_close(self):
        from factorcode.data.yfinance_source import load_snapshot
        with tempfile.TemporaryDirectory() as folder:
            market=load_snapshot(self.fixture(folder),'2026-04-02')
        self.assertEqual(market['calendar_status'],'XNYS verified')

    def test_midweek_signal_is_rejected(self):
        from factorcode.data.yfinance_source import load_snapshot
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'final market session'):
                load_snapshot(self.fixture(folder),'2026-04-01')

if __name__ == '__main__': unittest.main()
