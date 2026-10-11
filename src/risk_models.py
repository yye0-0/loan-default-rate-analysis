"""Bounded local modeling of Spark outputs; joint monthly competing hazards.

Do not claim distributed Cox or distributed multinomial model training.
"""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax,expit,logit
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from panel import FEATURES,cumulative_incidence
from metrics import CensorKM,horizon_target,evaluate
from spark_pipeline import read_local

def validate_config(config):
    phases=config['phases'];ordered=['train','calibration','validation','test']
    previous=None
    for phase in ordered:
        c=phases[phase];start,end,asof=[pd.Timestamp(c[k]) for k in ['start','end','asof']]
        if start>end or end>asof or (previous is not None and start<=previous):raise ValueError('Invalid phase dates')
        previous=end
    if pd.Timestamp(phases['train']['asof'])>=pd.Timestamp(phases['calibration']['start']):raise ValueError('Future training outcomes')
    if max(pd.Timestamp(phases[k]['asof']) for k in ['train','calibration','validation'])>=pd.Timestamp(phases['test']['start']):raise ValueError('Test starts before model freeze')
    if config['horizons']!=[12,24]:raise ValueError('Use 12/24 horizons for this experiment')
def design(frame,monthly=False):
    x=frame[FEATURES].astype(float).copy();x.loan_amount=np.log(x.loan_amount.clip(lower=1))
    if monthly:x['month_band']=pd.cut(frame.t,[0,3,6,12,24],labels=['1-3','4-6','7-12','13-24']).astype(str)
    return x
def prep(cols,monthly=False):
    numeric=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())])
    specs=[('numeric',numeric,cols)]
    if monthly:specs.append(('month',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['month_band']))
    return ColumnTransformer(specs)
def hazards(model,frame,theta=None):
    raw=model.predict_proba(design(frame,True));p=np.full((len(frame),5),1e-12)
    p[:,model.classes_.astype(int)]=raw;p/=p.sum(axis=1,keepdims=True)
    if theta is not None:
        theta=np.asarray(theta);p=softmax(np.log(np.clip(p,1e-12,1))/np.exp(theta[0])+np.r_[0.,theta[1:]],axis=1)
    return p
def predict_cif(model,frame,horizons=(12,24),theta=None):
    survival=np.ones(len(frame));cif=np.zeros((len(frame),4));result={}
    for t in range(1,max(horizons)+1):
        x=frame.copy();x['t']=t;h=hazards(model,x,theta)
        cif+=survival[:,None]*h[:,1:];survival*=h[:,0]
        if t in horizons:result[t]=np.column_stack([survival,cif.copy()])
    for values in result.values():
        if not np.allclose(values.sum(axis=1),1,atol=1e-10):raise RuntimeError('Probability mass error')
    return result
def fit_joint_calibration(model,period):
    raw=hazards(model,period);logs=np.log(np.clip(raw,1e-12,1));y=period.outcome.to_numpy(int)
    def loss(theta):
        p=softmax(logs/np.exp(theta[0])+np.r_[0.,theta[1:]],axis=1)
        return float(-np.mean(np.log(np.maximum(p[np.arange(len(y)),y],1e-12)))+.002*np.sum(theta**2))
    fit=minimize(loss,np.zeros(5),method='L-BFGS-B',bounds=[(-2,2)]+[(-3,3)]*4)
    if not fit.success:raise RuntimeError('Joint calibration failed: '+str(fit.message))
    return fit.x,{'objective':float(fit.fun),'success':bool(fit.success),'parameters':fit.x.tolist(),
                 'fitted_on':'calibration phase only; monthly log loss + ridge .002'}
def fit_platt(y,p,w):
    keep=w>0;y=y[keep];z=logit(np.clip(p[keep],1e-6,1-1e-6));weights=w[keep]
    if len(np.unique(y))<2:return np.array([0.,0.])
    def loss(t):
        pred=expit(np.exp(t[0])*z+t[1]);nll=-(y*np.log(np.maximum(pred,1e-12))+(1-y)*np.log(np.maximum(1-pred,1e-12)))
        return float(np.average(nll,weights=weights)+.001*np.sum(t*t))
    fit=minimize(loss,np.zeros(2),method='L-BFGS-B',bounds=[(-3,3),(-6,6)])
    if not fit.success:raise RuntimeError('Platt fit failed')
    return fit.x
def calibrate(p,t):return expit(np.exp(t[0])*logit(np.clip(p,1e-6,1-1e-6))+t[1])
def main():
    p=argparse.ArgumentParser();p.add_argument('--spark-dir',required=True);p.add_argument('--config',required=True)
    p.add_argument('--output',required=True);p.add_argument('--catboost',action='store_true');a=p.parse_args()
    config=json.loads(Path(a.config).read_text());validate_config(config);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    labels={k:read_local(Path(a.spark_dir)/f'model_labels_{k}') for k in config['phases']}
    periods={k:read_local(Path(a.spark_dir)/f'period_{k}') for k in config['phases']}
    seen=set()
    for k,frame in labels.items():
        if frame.loan_id.duplicated().any() or len(frame)>6000 or len(frame)<30 or seen.intersection(frame.loan_id):raise ValueError('Invalid bounded phase IDs/count')
        seen.update(frame.loan_id)
        if len(periods[k])>144000 or periods[k].duplicated(['loan_id','t']).any():raise ValueError('Invalid period export')
    kms={k:CensorKM().fit(f) for k,f in labels.items()}
    train=periods['train'];joint=Pipeline([('preprocess',prep(FEATURES,True)),('model',LogisticRegression(max_iter=1800,C=.5,random_state=42))])
    joint.fit(design(train,True),train.outcome.astype(int));theta,calmeta=fit_joint_calibration(joint,periods['calibration'])
    # One joint curve selected across horizons using validation ONLY. Freeze before test.
    curve_scores={}
    for name,t in [('joint_raw',None),('joint_calibrated',theta)]:
        val=predict_cif(joint,labels['validation'],theta=t)
        curve_scores[name]=float(np.mean([evaluate(labels['validation'],val[h][:,1],h,kms['validation'])[0]['ipcw_brier'] for h in config['horizons']]))
    selected_curve=min(curve_scores,key=curve_scores.get);selected_theta=theta if selected_curve=='joint_calibrated' else None
    freeze={'selected_variant':selected_curve,'validation_mean_brier':curve_scores,'selected_on':'validation only; mean 12/24 IPCW Brier',
      'freeze_before':config['phases']['test']['start'],'data_kind':config['data_kind'],'joint_calibration':calmeta}
    (out/'frozen_joint_curve.json').write_text(json.dumps(freeze,indent=2))
    joblib.dump({'model':joint,'joint_calibration':selected_theta,'selected_variant':selected_curve,
       'data_kind':config['data_kind'],'features':FEATURES},out/'joint_model.joblib')
    report={'data_kind':config['data_kind'],'phases':{k:len(f) for k,f in labels.items()},'joint_curve':freeze,
      'horizons':{},'optional_models_not_evaluated':[],
      'limits':['Bounded local monthly classifier, not distributed training','Pooled independent-censoring assumption',
        'Conditional on current at first observation, not lifetime default','Calendar month labels; publication lag not simulated']}
    predictions={};joint_predictions={name:{k:predict_cif(joint,f,theta=t) for k,f in labels.items()} for name,t in [('joint_raw',None),('joint_calibrated',theta)]}
    for h in config['horizons']:
        y,w,support=horizon_target(labels['train'],h,kms['train'])
        if not support['supported']:raise ValueError('Train IPCW unsupported')
        keep=w>0;known=labels['train'].loc[keep]
        definitions={'fico_logistic':(['fico'],LogisticRegression(max_iter=1000,C=.5)),
           'full_logistic':(FEATURES,LogisticRegression(max_iter=1000,C=.5)),
           'hist_gradient_boosting':(FEATURES,HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,l2_regularization=3,random_state=42))}
        if a.catboost:
            try:
                from catboost import CatBoostClassifier
                from catboost_compat import SklearnCatBoostClassifier
                definitions['catboost']=(FEATURES,SklearnCatBoostClassifier(CatBoostClassifier(iterations=200,depth=4,verbose=False,thread_count=2,random_seed=42)))
            except Exception as e:report['optional_models_not_evaluated'].append({'model':'catboost','horizon':h,'reason':str(e)})
        allp={'constant':{k:np.full(len(f),float(np.average(y[keep],weights=w[keep]))) for k,f in labels.items()}}
        for name,(cols,estimator) in definitions.items():
            model=Pipeline([('preprocess',prep(cols)),('model',estimator)])
            try:model.fit(design(known),y[keep],model__sample_weight=w[keep])
            except Exception as e:
                if name!='catboost':raise
                report['optional_models_not_evaluated'].append({'model':name,'horizon':h,'reason':str(e)});continue
            probs={k:model.predict_proba(design(f))[:,1] for k,f in labels.items()}
            cy,cw,cs=horizon_target(labels['calibration'],h,kms['calibration'])
            if not cs['supported']:raise ValueError('Calibration IPCW unsupported')
            platt=fit_platt(cy,probs['calibration'],cw)
            allp[name]={k:calibrate(v,platt) for k,v in probs.items()}
            joblib.dump({'model':model,'platt':platt,'horizon':h,'data_kind':config['data_kind']},out/f'{name}_{h}m.joblib')
        for name in joint_predictions:allp[name]={k:v[h][:,1] for k,v in joint_predictions[name].items()}
        validations={name:evaluate(labels['validation'],v['validation'],h,kms['validation'])[0] for name,v in allp.items()}
        winner=min(validations,key=lambda n:validations[n]['ipcw_brier'])
        (out/f'frozen_choice_{h}m.json').write_text(json.dumps({'selected':winner,'validation':validations,'data_kind':config['data_kind']},indent=2))
        tests={}
        for name,probs in allp.items():
            metric,bins=evaluate(labels['test'],probs['test'],h,kms['test'],bootstrap=100 if name==winner else 0)
            tests[name]=metric;bins.to_csv(out/f'calibration_{name}_{h}m.csv',index=False)
            predictions[(name,h)]=probs['test']
        test=labels['test'][['loan_id','duration','event_type']].copy()
        for name,v in allp.items():test[name]=v['test']
        test.to_csv(out/f'test_predictions_{h}m.csv',index=False)
        report['horizons'][str(h)]={'selected_on_validation':winner,'validation':validations,'test':tests}
        sub=[]
        for field,groups in [('fico_band',pd.cut(labels['test'].fico,[0,650,700,850],include_lowest=True)),
                             ('entry_age_band',pd.cut(labels['test'].entry_age_months,[-1,0,3,1000]))]:
            for band in groups.dropna().unique():
                mask=(groups==band).to_numpy();f=labels['test'].loc[mask]
                if len(f)>=20:
                    metric,_=evaluate(f,allp[winner]['test'][mask],h,kms['test'])
                    sub.append({'group':field,'band':str(band),'n':len(f),**metric})
        pd.DataFrame(sub).to_csv(out/f'subgroups_{h}m.csv',index=False)
    crossing=[]
    for name,h in predictions:
        if h!=12 or (name,24) not in predictions:continue
        crossing.append({'model':name,'crossed_12_gt_24':int((predictions[(name,12)]>predictions[(name,24)]+1e-12).sum()),
                         'meaning':'Separate fixed-horizon baseline; use selected joint curve for coherent inference'})
    pd.DataFrame(crossing).to_csv(out/'horizon_crossing_audit.csv',index=False)
    for name,phases in joint_predictions.items():
        f=labels['test'][['loan_id']].copy()
        for h,arr in phases['test'].items():
            for k in range(5):f[f'{"event_free" if k==0 else "cif_"+str(k)}_{h}m']=arr[:,k]
            f[f'credit_stress_proxy_{h}m']=arr[:,1]+arr[:,3]
        f.to_csv(out/f'{name}_test_joint_probabilities.csv',index=False)
    sel=joint_predictions[selected_curve]['test']
    report['selected_joint_credit_stress_proxy']={str(h):evaluate(labels['test'],v[:,1]+v[:,3],h,kms['test'],causes=(1,3))[0] for h,v in sel.items()}
    for k,f in labels.items():cumulative_incidence(f).to_csv(out/f'empirical_incidence_{k}.csv',index=False)
    (out/'risk_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
