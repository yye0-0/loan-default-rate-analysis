"""Generate clearly synthetic data for functional verification, never resume results."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--n',type=int,default=2400)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    r=np.random.default_rng(42);n=a.n
    cohort=pd.to_datetime('2012-01-01')+pd.to_timedelta(r.integers(0,2200,n),unit='D')
    fico=r.uniform(550,820,n);dti=r.uniform(5,50,n);rate=r.uniform(4,18,n);amount=r.uniform(3000,35000,n)
    income=r.uniform(20000,150000,n);grade=np.array(list('ABCDEFG'))[np.clip(((820-fico)/45).astype(int),0,6)]
    probability=1/(1+np.exp(-(-2+.02*(680-fico)+.04*(dti-20))))
    tab=pd.DataFrame({'id':np.arange(n),'issueDate':cohort.strftime('%Y-%m-%d'),'loanAmnt':amount,
                     'term':r.choice([3,5],n),'interestRate':rate,'annualIncome':income,'dti':dti,
                     'ficoRangeLow':fico-5,'ficoRangeHigh':fico+5,'revolUtil':r.uniform(0,100,n),
                     'earliesCreditLine':(cohort-pd.to_timedelta(r.integers(800,8000,n),unit='D')).strftime('%Y-%m-%d'),
                     'grade':grade,'homeOwnership':r.choice(['RENT','OWN','MORTGAGE'],n),
                     'verificationStatus':r.choice(['0','1','2'],n),'purpose':r.choice(['0','1','2'],n),
                     'applicationType':r.choice(['0','1'],n),'isDefault':r.binomial(1,probability)})
    tab.to_csv(out/'tabular_synthetic.csv',index=False)
    orig=pd.DataFrame({'loan_id':[f'DEMO{i}' for i in range(n)],'cohort_date':cohort.strftime('%Y-%m-01'),
                       'fico':fico,'dti':dti,'ltv':r.uniform(40,98,n),'interest_rate':rate,'loan_amount':amount})
    orig.to_csv(out/'orig_synthetic.csv',index=False)
    rows=[]
    for i in range(n):
        event_t=max(3,int(36*np.exp(-1.5*(probability[i]-.2))*r.weibull(1.4)))
        payoff=int(r.exponential(80))+1;observed=int(r.integers(12,61))
        stop=min(event_t,payoff,observed)
        for t in range(1,stop+1):
            delin=3 if t==event_t and t<=payoff else (max(0,3-(event_t-t)) if t<event_t else 0)
            rows.append((f'DEMO{i}',t,delin,'payoff' if t==payoff else 'active'))
    pd.DataFrame(rows,columns=['loan_id','month_index','delinquency_months','exit_type']).to_csv(
        out/'monthly_synthetic.csv',index=False)
    print(f'SYNTHETIC: {n} loans, {len(rows)} loan-months. Not empirical credit-risk evidence.')

if __name__=='__main__':main()
