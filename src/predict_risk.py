"""Inference from the frozen, horizon-coherent joint model."""
import argparse
import joblib,numpy as np,pandas as pd
from panel import FEATURES
from risk_models import predict_cif
def predict(bundle,frame):
    required=['loan_id',*FEATURES]
    if set(required)-set(frame):raise ValueError('Missing inference features')
    if len(frame)>6000 or not len(frame) or frame.loan_id.duplicated().any():raise ValueError('Use unique IDs, batches 1..6000')
    x=frame.copy()
    for c in FEATURES:x[c]=pd.to_numeric(x[c],errors='raise')
    if (x.entry_age_months<0).any():raise ValueError('Entry age must be nonnegative')
    for c,(lo,hi) in {'fico':(300,850),'dti':(0,65),'ltv':(1,998),'interest_rate':(.001,100),'loan_amount':(1,1e10)}.items():
        x.loc[~x[c].between(lo,hi),c]=np.nan
    probs=predict_cif(bundle['model'],x,theta=bundle['joint_calibration']);out=x[['loan_id']].copy()
    if np.any(probs[12][:,1:]>probs[24][:,1:]+1e-12):raise RuntimeError('Nonmonotone CIF')
    for h,values in probs.items():
        for k in range(5):out[f'{"event_free" if k==0 else "cif_"+str(k)}_{h}m']=values[:,k]
        out[f'credit_stress_proxy_{h}m']=values[:,1]+values[:,3]
    out['data_kind']=bundle['data_kind'];out['selected_variant']=bundle['selected_variant'];return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--csv',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    predict(joblib.load(a.model),pd.read_csv(a.csv,dtype={'loan_id':str})).to_csv(a.output,index=False)
