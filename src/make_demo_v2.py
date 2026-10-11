"""Seeded synthetic histories; never interpret scores as real performance."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit

CONFIG={'phases':{
 'train':{'start':'2007-01-01','end':'2015-12-01','asof':'2015-12-01'},
 'calibration':{'start':'2016-01-01','end':'2016-12-01','asof':'2019-12-01'},
 'validation':{'start':'2017-01-01','end':'2017-12-01','asof':'2019-12-01'},
 'test':{'start':'2020-01-01','end':'2021-12-01','asof':'2023-12-01'}},
 'horizons':[12,24],'data_kind':'SYNTHETIC_FUNCTIONAL_TEST'}
def generate(output,n=6000):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(42)
    months=pd.period_range('2007-01','2021-12',freq='M');orig=[];monthly=[]
    for i in range(n):
        lid=f'S{i:06d}';snap=months[int(rng.integers(len(months)))];age=int(rng.integers(0,7))
        fico=float(np.clip(rng.normal(700,55),400,840));dti=float(rng.uniform(10,60));ltv=float(rng.uniform(45,100))
        rate=float(rng.uniform(3,9));amount=float(rng.uniform(80000,500000))
        orig.append([lid,str((snap-age).start_time.date()),fico,dti,ltv,rate,amount])
        rows=[];limit=int(rng.integers(12,49));hasgap=rng.random()<.015;unknown=rng.random()<.012
        for t in range(limit+1):
            date=snap+t
            if date>pd.Period('2023-12','M'):break
            if hasgap and t==8:continue
            dpd='0';code='';mod='N'
            if unknown and t==7:dpd='XX'
            if t>0 and dpd=='0':
                stress=(700-fico)/65+(dti-30)/25+(ltv-80)/35
                macro=.7 if date.year in {2009,2020} else 0
                p1=.0015+.026*expit(-2+stress+macro)
                p2=.012+.020*expit((rate-6)+t/35)
                u=rng.random()
                if u<p1:dpd='3'
                elif u<p1+p2:code='01'
                elif u<p1+p2+.0013:dpd='RA';code=str(rng.choice(['02','03','09']))
                elif u<p1+p2+.0043:code=str(rng.choice(['15','16','96']))
                elif rng.random()<.035:dpd=str(rng.choice(['1','2']))
            rows.append([lid,str(date.start_time.date()),dpd,code,mod])
            if dpd=='3':
                for j in range(1,5):
                    if date+j<=pd.Period('2023-12','M'):
                        rows.append([lid,str((date+j).start_time.date()),'RA' if j==4 else '3','09' if j==4 else '', 'Y' if j==4 else 'N'])
                break
            if code:break
        monthly.extend(rows)
    o=pd.DataFrame(orig,columns=['loan_id','first_payment_month','fico','dti','ltv','interest_rate','loan_amount'])
    m=pd.DataFrame(monthly,columns=['loan_id','report_month','dpd','zero_code','modification'])
    o.to_csv(out/'orig.csv',index=False);m.to_csv(out/'monthly.csv',index=False)
    (out/'config.json').write_text(json.dumps(CONFIG,indent=2))
    # Adapter fixtures follow the July 2026 physical widths, not real records.
    with (out/'freddie_orig_synthetic.txt').open('w') as f:
        for r in o.to_dict('records'):
            a=['']*31
            for k,v in {0:r['fico'],1:r['first_payment_month'][:7].replace('-',''),9:r['dti'],10:r['loan_amount'],11:r['ltv'],12:r['interest_rate'],19:r['loan_id']}.items():a[k]=str(v)
            f.write('|'.join(a)+'\n')
    with (out/'freddie_perf_synthetic.txt').open('w') as f:
        for r in m.to_dict('records'):
            a=['']*35
            for k,v in {0:r['loan_id'],1:r['report_month'][:7].replace('-',''),3:r['dpd'],7:r['modification'],8:r['zero_code']}.items():a[k]=str(v)
            f.write('|'.join(a)+'\n')
    return {'loans':len(o),'monthly_rows':len(m),'data_kind':CONFIG['data_kind']}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--n',type=int,default=6000)
    a=p.parse_args();print(json.dumps(generate(a.output,a.n)))
