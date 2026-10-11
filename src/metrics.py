"""IPCW Brier/AUC/calibration. Administrative censoring AFTER month t.

The pooled censoring KM assumes independent censoring. Missingness can be
informative; this is a diagnostic benchmark, not an identification guarantee.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
class CensorKM:
    def fit(self,labels):
        self.times=[];self.values=[];g=1.
        for t in sorted(labels.duration.unique()):
            risk=int((labels.duration>=t).sum());end=labels[labels.duration==t]
            censor=int((end.event_type==0).sum());terminal=len(end)-censor
            # Terminal events occur before end-of-month administrative censoring.
            denominator=risk-terminal
            if denominator:g*=1-censor/denominator
            self.times.append(t);self.values.append(g)
        return self
    def before(self,t):
        idx=np.searchsorted(self.times,t,side='left')-1
        return np.where(idx<0,1.,np.asarray(self.values)[np.maximum(idx,0)])
def horizon_target(labels,h,km,causes=(1,)):
    d=labels.duration.to_numpy();e=labels.event_type.to_numpy();terminal=(e>0)&(d<=h)
    known=terminal|(d>=h);y=(terminal&np.isin(e,causes)).astype(int)
    g=km.before(np.where(terminal,d,h));weights=np.where(known,1/np.maximum(g,1e-12),0.)
    supported=bool(np.all(g[known]>=.05))
    return y,weights,{'known':int(known.sum()),'unknown':int((~known).sum()),'minimum_G':float(g[known].min()) if known.any() else 0.,'supported':supported}
def evaluate(labels,p,h,km,causes=(1,),bootstrap=0):
    y,w,s=horizon_target(labels,h,km,causes);p=np.asarray(p);mask=w>0
    if not s['supported']:raise ValueError('Unsupported IPCW horizon: G < .05')
    result={**s,'ipcw_brier':float(np.sum(w*(y-p)**2)/len(y)),
       'complete_case_brier':float(np.mean((y[mask]-p[mask])**2)),
       'roc_auc':float(roc_auc_score(y[mask],p[mask],sample_weight=w[mask])) if len(np.unique(y[mask]))==2 else None,
       'average_precision':float(average_precision_score(y[mask],p[mask],sample_weight=w[mask])) if y[mask].sum() else None}
    k=max(1,int(np.ceil(.05*len(y))));chosen=np.argsort(-p,kind='stable')[:k]
    result['top_5pct_weighted_event_capture']=float(np.sum(w[chosen]*y[chosen])/max(np.sum(w*y),1e-12))
    if bootstrap:
        rng=np.random.default_rng(42);samples=[]
        for _ in range(bootstrap):
            ix=rng.integers(0,len(y),len(y));samples.append(float(np.mean(w[ix]*(y[ix]-p[ix])**2)))
        result['brier_bootstrap_95pct']=np.quantile(samples,[.025,.975]).tolist()
        result['bootstrap_note']='Loan resampling; fixed predictions and censoring model; training uncertainty excluded'
    bins=pd.DataFrame({'p':p[mask],'y':y[mask],'w':w[mask]})
    bins['bin']=pd.qcut(bins.p,10,duplicates='drop') if bins.p.nunique()>1 else 'constant'
    rows=[]
    for b,g in bins.groupby('bin',observed=True):
        rows.append({'bin':str(b),'n':len(g),'mean_predicted':float(np.average(g.p,weights=g.w)),
          'weighted_observed':float(np.average(g.y,weights=g.w))})
    return result,pd.DataFrame(rows)
