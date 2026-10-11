"""Reproduce full synthetic label and saved-model integration verification."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,joblib
from panel import label_asof,FEATURES
from spark_pipeline import read_local
from predict_risk import predict
def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--require-catboost',action='store_true');a=p.parse_args();r=Path(a.results);d=r/'synthetic_inputs'
    o=pd.read_csv(d/'orig.csv',dtype={'loan_id':str})
    m=pd.read_csv(d/'monthly.csv',keep_default_na=False,dtype={'loan_id':str,'dpd':str,'zero_code':str,'modification':str})
    if len(o)>6000:raise ValueError('Full oracle verification is bounded to 6000 synthetic loans')
    config=json.loads((d/'config.json').read_text());report={'independent_full_labels':{}}
    cols=['loan_id','duration','event_type','entry_age_months','stop_reason']
    for phase,c in config['phases'].items():
        expected,_=label_asof(o,m,c['asof']);dates=pd.to_datetime(expected.snapshot_month)
        expected=expected[dates.between(c['start'],c['end'])][cols].sort_values('loan_id').reset_index(drop=True)
        actual=read_local(r/'spark'/f'labels_{phase}')[cols].sort_values('loan_id').reset_index(drop=True)
        pd.testing.assert_frame_equal(expected,actual,check_dtype=False)
        report['independent_full_labels'][phase]={'loans':len(expected),'exact_match':True}
    test=read_local(r/'spark/model_labels_test');inputs=test[['loan_id',*FEATURES]]
    inputs.head(10).to_csv(r/'risk/inference_example.csv',index=False)
    values=predict(joblib.load(r/'risk/joint_model.joblib'),inputs)
    values.to_csv(r/'risk/inference_verified_all_test.csv',index=False)
    for h in [12,24]:
        cols=[f'event_free_{h}m',*[f'cif_{i}_{h}m' for i in range(1,5)]]
        np.testing.assert_allclose(values[cols].sum(axis=1),1,atol=1e-10)
    for i in range(1,5):
        if not (values[f'cif_{i}_24m']>=values[f'cif_{i}_12m']-1e-12).all():raise ValueError('CIF horizon crossing')
    selected=values.selected_variant.iloc[0]
    original=pd.read_csv(r/'risk'/f'{selected}_test_joint_probabilities.csv').set_index('loan_id')
    probs=[c for c in original.columns if c!='loan_id']
    np.testing.assert_allclose(values.set_index('loan_id').sort_index()[probs],original.sort_index()[probs],atol=1e-10)
    report['saved_model_inference']={'loans':len(values),'mass_conservation':True,'all_four_cifs_monotone':True,
                                   'agrees_with_original_test_predictions':True,'selected_variant':selected}
    if a.require_catboost:
        from verification_gate import require_catboost
        from risk_models import design,calibrate
        require_catboost(json.loads((r/'risk/risk_report.json').read_text()),r/'risk')
        for h in [12,24]:
            bundle=joblib.load(r/'risk'/f'catboost_{h}m.joblib')
            native=bundle['model'].named_steps['model'].model_
            if type(native).__name__!='CatBoostClassifier':raise ValueError('Expected native CatBoost model')
            actual=calibrate(bundle['model'].predict_proba(design(test))[:,1],bundle['platt'])
            expected=pd.read_csv(r/'risk'/f'test_predictions_{h}m.csv',dtype={'loan_id':str}).set_index('loan_id').loc[test.loan_id,'catboost'].to_numpy()
            np.testing.assert_allclose(actual,expected,atol=1e-10)
        report['saved_catboost_inference']={'loans':len(test),'horizons':[12,24],
            'native_engine':'CatBoostClassifier','agrees_with_original_test_predictions':True}
    (r/'integration_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
