"""Bounded lifelines Cox + PH diagnostics, auxiliary cause-specific analysis."""
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index
from panel import FEATURES
from spark_pipeline import read_local
from risk_models import design
def main():
    p=argparse.ArgumentParser();p.add_argument('--spark-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    train=read_local(Path(a.spark_dir)/'model_labels_train');test=read_local(Path(a.spark_dir)/'model_labels_test')
    pre=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler())
    f=pd.DataFrame(pre.fit_transform(design(train)),columns=FEATURES)
    f['duration']=train.duration.to_numpy();f['event']=(train.event_type==1).to_numpy(int)
    model=CoxPHFitter(penalizer=.05).fit(f,duration_col='duration',event_col='event')
    model.summary.to_csv(out/'cox_coefficients.csv');ph=proportional_hazard_test(model,f,time_transform='rank')
    ph.summary.to_csv(out/'ph_diagnostics.csv');x=pd.DataFrame(pre.transform(design(test)),columns=FEATURES)
    surv=model.predict_survival_function(x,times=[12,24]);pred=test[['loan_id','duration','event_type']].copy()
    for h in [12,24]:pred[f'net_risk_{h}m']=1-surv.loc[h].to_numpy()
    pred.to_csv(out/'cox_net_risk_test.csv',index=False)
    c=concordance_index(test.duration.to_numpy(),-model.predict_partial_hazard(x).to_numpy(),(test.event_type==1).to_numpy())
    report={'engine':'lifelines (local, bounded 6000 per phase)','train_n':len(train),'test_n':len(test),
            'unweighted_test_harrell_c':float(c),'ph_violations_p_lt_01':ph.summary.index[ph.summary.p<.01].tolist(),
            'estimand':'Cause-specific Cox/net risk, not competing-risk probability',
            'diagnostic_note':'PH p-values retained without claiming assumption is true; penalized fitted model.'}
    joblib.dump({'preprocessor':pre,'model':model},out/'cox_model.joblib');(out/'cox_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
