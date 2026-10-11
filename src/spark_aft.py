"""Spark MLlib Weibull AFT, auxiliary net risk (NOT competing-risk CIF)."""
import argparse,json,time,shutil
from pathlib import Path
from spark_pipeline import make_spark,dataset_parts,write_parquet
from panel import FEATURES
def main():
    from pyspark.sql import functions as F
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import Imputer,VectorAssembler,StandardScaler
    from pyspark.ml.regression import AFTSurvivalRegression
    p=argparse.ArgumentParser();p.add_argument('--spark-dir',required=True);p.add_argument('--output',required=True)
    p.add_argument('--master',default='local[2]');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    s=make_spark(a.master);start=time.perf_counter();frames={}
    for name in ['train','validation','test']:
        f=s.read.parquet(*dataset_parts(Path(a.spark_dir)/f'labels_{name}')).withColumn('loan_amount',F.log('loan_amount'))
        frames[name]=f.withColumn('event',(F.col('event_type')==1).cast('double')).withColumn('duration',F.col('duration').cast('double'))
    imputed=[c+'_imputed' for c in FEATURES]
    pipeline=Pipeline(stages=[Imputer(inputCols=FEATURES,outputCols=imputed,strategy='median'),
        VectorAssembler(inputCols=imputed,outputCol='raw_features'),
        StandardScaler(inputCol='raw_features',outputCol='features',withMean=True,withStd=True),
        AFTSurvivalRegression(labelCol='duration',censorCol='event',featuresCol='features',maxIter=100,tol=1e-6)])
    model=pipeline.fit(frames['train']);sigma=float(model.stages[-1].scale)
    if (out/'model').exists():shutil.rmtree(out/'model')
    model.write().overwrite().save(str(out/'model'))
    for name in ['validation','test']:
        pred=model.transform(frames[name])
        for h in [12,24]:pred=pred.withColumn(f'net_risk_{h}m',1-F.exp(-F.pow(F.lit(h)/F.col('prediction'),1/sigma)))
        write_parquet(pred.select('loan_id','duration','event_type','prediction','net_risk_12m','net_risk_24m'),out/f'{name}_predictions')
    meta={'engine':'Spark MLlib AFTSurvivalRegression','spark_version':s.version,'master':s.sparkContext.master,
          'scale_sigma':sigma,'weibull_shape':1/sigma,'elapsed_seconds':time.perf_counter()-start,
          'split_counts':{k:f.count() for k,f in frames.items()},
          'estimand':'Cause-specific Weibull net risk; competing events censored. 1-S is not cumulative incidence.'}
    (out/'aft_report.json').write_text(json.dumps(meta,indent=2));s.stop();print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
