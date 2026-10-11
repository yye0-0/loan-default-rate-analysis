"""Original tabular branch. Fit preprocessing only on earlier issue dates."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (roc_auc_score,average_precision_score,brier_score_loss,
                             accuracy_score,precision_score,recall_score,f1_score)
from features import build_features,temporal_split,NUMERIC,CATEGORICAL

def evaluate(y,p,threshold):
    pred = p>=threshold
    return {'roc_auc':float(roc_auc_score(y,p)),
            'average_precision':float(average_precision_score(y,p)),
            'brier':float(brier_score_loss(y,p)),
            'accuracy':float(accuracy_score(y,pred)),
            'precision':float(precision_score(y,pred,zero_division=0)),
            'recall':float(recall_score(y,pred,zero_division=0)),
            'f1':float(f1_score(y,pred,zero_division=0))}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv',required=True)
    p.add_argument('--train-end',required=True)
    p.add_argument('--valid-end',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--catboost',action='store_true')
    p.add_argument('--feature-set',choices=['base','add_composite','add_grade_proxy','add_both'],default='base')
    p.add_argument('--ablation',action='store_true')
    p.add_argument('--validation-only',action='store_true')
    p.add_argument('--synthetic',action='store_true')
    args = p.parse_args()
    if args.ablation:
        import subprocess,sys
        out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
        def batch(validation_only):
            runs={}
            for feature_set in ['base','add_composite','add_grade_proxy','add_both']:
                dest=out/('validation_selection' if validation_only else 'evaluated')/feature_set
                cmd=[sys.executable,__file__,'--csv',str(Path(args.csv).resolve()),'--train-end',args.train_end,'--valid-end',args.valid_end,'--output',str(dest.resolve()),'--feature-set',feature_set]
                if validation_only:cmd+=['--validation-only']
                if args.synthetic:cmd+=['--synthetic']
                if args.catboost:cmd+=['--catboost']
                result=subprocess.run(cmd,capture_output=True,text=True)
                (out/(('selection_' if validation_only else 'test_')+feature_set+'.log')).write_text(result.stdout+'\n'+result.stderr)
                if result.returncode:raise RuntimeError('Ablation failed: '+feature_set)
                runs[feature_set]=json.loads((dest/'classification_metrics.json').read_text())
            return runs
        selection=batch(True)
        candidates=[(v['validation']['brier'],k,n) for k,r in selection.items() for n,v in r['models'].items()]
        _,selected,selected_model=min(candidates)
        freeze={'selected_feature_set':selected,'selected_model':selected_model,'selection':'validation Brier only; frozen before test evaluation'}
        (out/'frozen_ablation_choice.json').write_text(json.dumps(freeze,indent=2))
        runs=batch(False)
        (out/'ablation_report.json').write_text(json.dumps({'selected_feature_set_on_validation':selected,'selected_model':selected_model,'freeze':freeze,'runs':runs,'limit':'Final-status label has unknown availability; retrospective cohort holdout, not fixed-horizon deployment PD'},indent=2))
        return
    raw = pd.read_csv(args.csv)
    if raw.id.duplicated().any():
        raise ValueError('Duplicate loan IDs; resolve explicitly before splitting')
    y = pd.to_numeric(raw.isDefault,errors='raise').astype(int)
    if not y.isin([0,1]).all():
        raise ValueError('isDefault must be 0/1')
    X = build_features(raw)
    masks = temporal_split(raw,args.train_end,args.valid_end)
    for m in masks:
        if y[m].nunique()!=2:
            raise ValueError('Each split must have both classes')
    numeric=[c for c in NUMERIC if c not in {'risk_composite','capm_grade_proxy'}]
    if args.feature_set in {'add_composite','add_both'}:numeric+=['risk_composite']
    if args.feature_set in {'add_grade_proxy','add_both'}:numeric+=['capm_grade_proxy']
    pre = ColumnTransformer([
        ('num',Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
                         ('scale',StandardScaler())]),numeric),
        ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CATEGORICAL)])
    matrices = [pre.fit_transform(X[masks[0]])]
    matrices.extend(pre.transform(X[m]) for m in masks[1:])
    labels = [y[m].to_numpy() for m in masks]
    models = {'logistic':LogisticRegression(max_iter=1500,random_state=42),
              'random_forest':RandomForestClassifier(n_estimators=120,max_depth=10,
                    min_samples_leaf=15,n_jobs=2,random_state=42)}
    optional=[]
    if args.catboost:
        try:
            from catboost import CatBoostClassifier
            models['catboost'] = CatBoostClassifier(iterations=400,depth=5,learning_rate=.05,
                        loss_function='Logloss',verbose=False,thread_count=2,random_seed=42)
        except Exception as e:optional.append({'model':'catboost','reason':str(e)})
    out = Path(args.output);out.mkdir(parents=True,exist_ok=True)
    metrics = {'data_kind':'SYNTHETIC_SMOKE_TEST' if args.synthetic else 'USER_SUPPLIED_DATA',
               'split_counts':[int(m.sum()) for m in masks],
               'train_end':args.train_end,'valid_end':args.valid_end,
               'target':'source final-status isDefault; horizon unknown','feature_set':args.feature_set,'numeric_features':numeric,'optional_models_not_evaluated':optional,'models':{}}
    for name,model in models.items():
        if name=='catboost':
            try:model.fit(matrices[0],labels[0],eval_set=(matrices[1],labels[1]),early_stopping_rounds=30)
            except Exception as e:
                optional.append({'model':'catboost','reason':str(e)});continue
        else:
            model.fit(matrices[0],labels[0])
        pv = model.predict_proba(matrices[1])[:,1]
        # Tune threshold on validation only. Reuse once on the held-out test set.
        grid = np.linspace(.02,.8,80)
        threshold = float(max(grid,key=lambda t:f1_score(labels[1],pv>=t,zero_division=0)))
        if args.validation_only:
            metrics['models'][name]={'threshold':threshold,'validation':evaluate(labels[1],pv,threshold)}
            continue
        pt = model.predict_proba(matrices[2])[:,1]
        metrics['models'][name] = {'threshold':threshold,'validation':evaluate(labels[1],pv,threshold),
                                  'test':evaluate(labels[2],pt,threshold)}
        joblib.dump({'preprocessor':pre,'model':model,'threshold':threshold},out/f'{name}.joblib')
        if hasattr(model,'feature_importances_'):
            pd.DataFrame({'feature':pre.get_feature_names_out(),
                          'importance':model.feature_importances_}).sort_values('importance',ascending=False)\
                .to_csv(out/f'{name}_feature_importance.csv',index=False)
        # Illustrative 300-850 scale from the original idea; no external FICO equivalence.
        pd.DataFrame({'id':raw.loc[masks[2],'id'],'pd':pt,'illustrative_score':850-550*pt}).to_csv(
             out/f'{name}_test_scores.csv',index=False)
    # Fit clusters on train only. Report actual variance, never assume 5 PCs retain 95%.
    num = matrices[0][:,:len(numeric)]
    pca = PCA(n_components=.95,svd_solver='full').fit(num)
    kmeans = KMeans(n_clusters=4,n_init=10,random_state=42).fit(pca.transform(num))
    metrics['pca'] = {'components':int(pca.n_components_),
                     'retained_variance':float(pca.explained_variance_ratio_.sum())}
    if not args.validation_only:
        pd.DataFrame({'id':raw.loc[masks[2],'id'],
                      'cluster':kmeans.predict(pca.transform(matrices[2][:,:len(numeric)])),
                      'isDefault':labels[2]}).to_csv(out/'cluster_test.csv',index=False)
    joblib.dump({'pca':pca,'kmeans':kmeans},out/'segmentation.joblib')
    (out/'classification_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':
    main()
