"""统一周研究入口；通过配置选择日期和权重，不连接交易账户。"""
from pathlib import Path
import sys,os,argparse
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/weekly/.plot-cache'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'outputs/weekly/.plot-cache'))
os.environ.setdefault('MPLBACKEND','Agg')
from factorcode.weekly_config import load_config
from factorcode.weekly_workflow import run

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--stage',choices=['download','research','all'],default='research')
    parser.add_argument('--study-weights',action='store_true')
    parser.add_argument('--open-chart',action='store_true')
    parser.add_argument('--publish',action='store_true',help='Export public fields and normalized chart to week directory; no Git push')
    args=parser.parse_args()
    try: out=run(load_config(args.config),args.stage,args.study_weights,args.open_chart,args.publish)
    except (ValueError,RuntimeError,OSError,KeyError) as exc: parser.exit(2,f'Error: {exc}\n')
    print('Results:',out)

if __name__=='__main__': main()
