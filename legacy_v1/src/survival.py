"""AFT prototype runnable with scipy; optional lifelines Cox reference.

Cause-specific analysis censors payoffs. 1-S is a net-risk estimate, not the
real-world probability of default in the presence of prepayment. Use the
Aalen-Johansen output for unconditional empirical cumulative incidence.
"""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from panel import build_labels,cumulative_incidence,calendar_hazard

FEATURES = ['fico','dti','ltv','interest_rate','loan_amount']

def fit_weibull(X,time,event):
    design = np.column_stack([np.ones(len(X)),X])
    log_t = np.log(time)
    def loss(theta):
        shape = np.exp(theta[-1])
        z = (log_t-design@theta[:-1])*shape
        # log f(t) = log(shape)-log(t)+z-exp(z)
        return float(np.mean(np.exp(np.clip(z,-50,50))-event*(theta[-1]-log_t+z))
                     + 1e-4*np.sum(theta[1:-1]**2))
    initial = np.zeros(design.shape[1]+1);initial[0] = np.median(log_t)
    fitted = minimize(loss,initial,method='L-BFGS-B',bounds=[(None,None)]*design.shape[1]+[(-3,3)])
    if not fitted.success:
        raise RuntimeError(f'Weibull fit failed: {fitted.message}')
    return fitted.x

def concordance(time,event,predicted_time):
    # Small-sample Harrell C, O(n^2). Caller limits evaluation sample and records size.
    wins,pairs = 0.,0
    for i in range(len(time)):
        if event[i]:
            js = time>time[i]
            pairs += int(js.sum())
            wins += float((predicted_time[i]<predicted_time[js]).sum())
            wins += .5*float((predicted_time[i]==predicted_time[js]).sum())
    return wins/pairs if pairs else None

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--orig',required=True);p.add_argument('--monthly',required=True)
    p.add_argument('--train-end',required=True);p.add_argument('--valid-end',required=True)
    p.add_argument('--output',required=True);p.add_argument('--cox',action='store_true')
    p.add_argument('--synthetic',action='store_true')
    a=p.parse_args()
    orig=pd.read_csv(a.orig,dtype={'loan_id':str})
    monthly=pd.read_csv(a.monthly,dtype={'loan_id':str})
    labels,excluded=build_labels(orig,monthly)
    dates=pd.to_datetime(labels.cohort_date,errors='raise')
    t1,t2=pd.Timestamp(a.train_end),pd.Timestamp(a.valid_end)
    if t1>=t2:raise ValueError('train_end must precede valid_end')
    train,valid,test=labels[dates<=t1],labels[(dates>t1)&(dates<=t2)],labels[dates>t2]
    if min(len(train),len(valid),len(test))<10 or train.event.sum()==0:
        raise ValueError('Insufficient observations/events in time splits')
    prep=make_pipeline(SimpleImputer(strategy='median'),StandardScaler())
    X=prep.fit_transform(train[FEATURES])
    theta=fit_weibull(X,train.duration.to_numpy(),train.event.to_numpy())
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    result={'data_kind':'SYNTHETIC_SMOKE_TEST' if a.synthetic else 'USER_SUPPLIED_DATA',
            'event':'first observed 90+ day delinquency (delinquency_months >= 3)',
            'estimand':'cause-specific/net risk, payoff treated as censoring',
            'included':len(labels),'excluded':len(excluded),'split_counts':[len(train),len(valid),len(test)],
            'shape':float(np.exp(theta[-1])),'concordance':{}}
    for name,frame in [('validation',valid),('test',test)]:
        design=np.column_stack([np.ones(len(frame)),prep.transform(frame[FEATURES])])
        scale=np.exp(np.clip(design@theta[:-1],-30,30))
        median=scale*np.log(2)**(1/np.exp(theta[-1]))
        pred=frame[['loan_id','duration','event_type']].copy()
        pred['predicted_median_months']=median
        for h in [6,12,24]:
            pred[f'net_risk_{h}m']=1-np.exp(-(h/scale)**np.exp(theta[-1]))
        pred.to_csv(out/f'aft_{name}.csv',index=False)
        sample=frame.sample(min(2000,len(frame)),random_state=42)
        pos=frame.index.get_indexer(sample.index)
        result['concordance'][name]={'harrell_c':concordance(sample.duration.to_numpy(),sample.event.to_numpy(),median[pos]),
                                     'evaluated_n':len(sample),'method':'Harrell C, unweighted, no IPCW'}
    if a.cox:
        from lifelines import CoxPHFitter
        cox_train=pd.DataFrame(X,columns=FEATURES,index=train.index)
        cox_train['duration']=train.duration;cox_train['event']=train.event
        cox=CoxPHFitter(penalizer=.05).fit(cox_train,duration_col='duration',event_col='event')
        cox.summary.to_csv(out/'cox_coefficients.csv')
        cox.check_assumptions(cox_train,p_value_threshold=.01,show_plots=False)
        ztest=pd.DataFrame(prep.transform(test[FEATURES]),columns=FEATURES,index=test.index)
        med=cox.predict_partial_hazard(ztest).to_numpy()
        sample_idx=np.random.default_rng(42).choice(len(test),min(2000,len(test)),replace=False)
        result['cox_test_harrell_c']=concordance(test.duration.to_numpy()[sample_idx],
                                  test.event.to_numpy()[sample_idx],-med[sample_idx])
        joblib.dump(cox,out/'cox.joblib')
    joblib.dump({'preprocessor':prep,'theta':theta,'features':FEATURES},out/'weibull_aft.joblib')
    labels.to_csv(out/'labels.csv',index=False);excluded.to_csv(out/'excluded.csv',index=False)
    cumulative_incidence(labels).to_csv(out/'descriptive_cumulative_incidence.csv',index=False)
    calendar_hazard(orig,monthly).to_csv(out/'calendar_hazard.csv',index=False)
    (out/'survival_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
