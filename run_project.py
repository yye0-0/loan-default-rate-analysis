"""Single reproducible entry point; exit nonzero when a required stage fails."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--demo',action='store_true');p.add_argument('--n',type=int,default=6000)
    p.add_argument('--orig');p.add_argument('--monthly');p.add_argument('--config');p.add_argument('--output',default='results')
    p.add_argument('--master',default='local[2]');p.add_argument('--freddie-version',choices=['canonical','legacy','july2026'],default='canonical')
    p.add_argument('--diagnostics',action='store_true');p.add_argument('--forecast',action='store_true');p.add_argument('--catboost',action='store_true');a=p.parse_args()
    output=Path(a.output);output=output if output.is_absolute() else ROOT/output;output.mkdir(parents=True,exist_ok=True)
    logs=output/'logs';logs.mkdir(exist_ok=True);completed=[]
    def run(name,script,args):
        with (logs/(name+'.log')).open('w') as f:
            result=subprocess.run([sys.executable,str(ROOT/'src'/script),*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Stage failed: '+name+'; inspect '+str(logs/(name+'.log')))
        completed.append(name);print('COMPLETED',name,flush=True)
    if a.demo:
        if a.freddie_version!='canonical':raise ValueError('Demo emits canonical inputs; adapter verified separately')
        data=output/'synthetic_inputs';run('generate','make_demo_v2.py',['--n',a.n,'--output',data]);orig=data/'orig.csv';monthly=data/'monthly.csv';config=data/'config.json'
    else:
        if not all([a.orig,a.monthly,a.config]):raise ValueError('Supply --orig --monthly --config or --demo')
        orig,monthly,config=[Path(x).resolve() for x in [a.orig,a.monthly,a.config]]
    sys.path.insert(0,str(ROOT/'src'))
    from risk_models import validate_config
    c=json.loads(config.read_text());validate_config(c)
    run('spark_etl','spark_pipeline.py',['--orig',orig,'--monthly',monthly,'--config',config,'--output',output/'spark','--master',a.master,'--freddie-version',a.freddie_version])
    run('risk_models','risk_models.py',['--spark-dir',output/'spark','--config',config,'--output',output/'risk']+(['--catboost'] if a.catboost else []))
    if a.diagnostics:
        run('spark_aft','spark_aft.py',['--spark-dir',output/'spark','--output',output/'aft','--master',a.master])
        run('cox','cox_diagnostics.py',['--spark-dir',output/'spark','--output',output/'cox'])
    if a.forecast:
        run('calendar','calendar_series.py',['--spark-dir',output/'spark','--config',config,'--output',output/'calendar','--master',a.master])
        run('forecast','forecast.py',['--csv',output/'calendar/calendar_hazard.csv','--output',output/'forecast'])
    (output/'run_complete.json').write_text(json.dumps({'data_kind':c['data_kind'],'completed':completed,'real_data_experiment':c['data_kind']!='SYNTHETIC_FUNCTIONAL_TEST'},indent=2))
if __name__=='__main__':main()
