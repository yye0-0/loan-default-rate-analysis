"""PySpark ETL + Spark Weibull AFT. Requires Java 11/17 and PySpark 3.5.

--freddie consumes headerless pipe-delimited Standard Dataset files, using the
published 2026 User Guide positions. Confirm field layout for your downloaded release.
Without --freddie, reads canonical CSV files described in README.
"""
import argparse,json,time
from pathlib import Path

def main():
    from pyspark.sql import SparkSession,functions as F,Window
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import VectorAssembler,StandardScaler
    from pyspark.ml.regression import AFTSurvivalRegression
    p=argparse.ArgumentParser()
    p.add_argument('--orig',required=True);p.add_argument('--monthly',required=True)
    p.add_argument('--output',required=True);p.add_argument('--train-end',required=True)
    p.add_argument('--valid-end',required=True);p.add_argument('--freddie',action='store_true')
    a=p.parse_args()
    if a.train_end>=a.valid_end:raise ValueError('Use ascending ISO date split boundaries')
    spark=SparkSession.builder.appName('LoanDefaultRebuild').getOrCreate()
    started=time.perf_counter()
    if a.freddie:
        ro=spark.read.option('sep','|').csv(a.orig)
        rm=spark.read.option('sep','|').csv(a.monthly)
        # 1-based vendor positions -> 0-based Spark columns.
        o=ro.select(F.trim('_c19').alias('loan_id'),
                    F.to_date('_c1','yyyyMM').alias('cohort_date'),
                    F.col('_c0').cast('double').alias('fico'),
                    F.col('_c9').cast('double').alias('dti'),
                    F.col('_c11').cast('double').alias('ltv'),
                    F.col('_c12').cast('double').alias('interest_rate'),
                    F.col('_c10').cast('double').alias('loan_amount'))
        m=rm.select(F.trim('_c0').alias('loan_id'),
                    F.to_date('_c1','yyyyMM').alias('report_date'),
                    F.trim('_c3').alias('delinquency_raw'),
                    F.trim('_c8').alias('zero_code'),
                    F.trim('_c7').alias('modification'))
        m=m.join(o.select('loan_id','cohort_date'),'loan_id','inner')
        m=m.withColumn('month_index',(F.months_between('report_date','cohort_date')+1).cast('int'))
        m=m.withColumn('delinquency_months',F.col('delinquency_raw').cast('int'))
        m=m.withColumn('invalid',F.col('delinquency_months').isNull() |
                       F.col('modification').isin('Y','P'))
        m=m.withColumn('exit_type',F.when(F.col('zero_code')=='01','payoff')
                  .when(F.coalesce(F.col('zero_code'),F.lit('')).isin('','00'),'active').otherwise('other'))
        # Acquisition-time FICO and DTI availability must be checked for your use case.
        o=o.filter(F.col('fico').between(300,850) & F.col('dti').between(0,65) &
                   F.col('ltv').between(1,998) & (F.col('interest_rate')>0) & (F.col('loan_amount')>0))
    else:
        o=spark.read.option('header',True).csv(a.orig)
        m=spark.read.option('header',True).csv(a.monthly)
        o=o.withColumn('cohort_date',F.to_date('cohort_date'))
        for c in ['fico','dti','ltv','interest_rate','loan_amount']:o=o.withColumn(c,F.col(c).cast('double'))
        for c in ['month_index','delinquency_months']:m=m.withColumn(c,F.col(c).cast('int'))
        m=m.withColumn('invalid',F.lit(False))
    o=o.select('loan_id','cohort_date','fico','dti','ltv','interest_rate','loan_amount').cache()
    m=m.select('loan_id','month_index','delinquency_months','exit_type','invalid').cache()
    if o.groupBy('loan_id').count().filter('count > 1').limit(1).count():
        raise ValueError('Duplicate origination loan IDs')
    if m.groupBy('loan_id','month_index').count().filter('count > 1').limit(1).count():
        raise ValueError('Duplicate loan-months')
    bad=F.col('loan_id').isNull() | F.col('exit_type').isNull() | F.col('month_index').isNull() | (F.col('month_index')<1) | F.col('delinquency_months').isNull() | \
        (F.col('delinquency_months')<0) | ~F.col('exit_type').isin('active','payoff','other') | F.col('invalid')
    invalid_ids=m.filter(bad).select('loan_id').distinct()
    m=m.join(invalid_ids,'loan_id','left_anti').join(o.select('loan_id'),'loan_id','inner')
    m=m.withColumn('row_event',F.when(F.col('delinquency_months')>=3,1)
                     .when(F.col('exit_type')=='payoff',2).otherwise(0))
    w=Window.partitionBy('loan_id')
    m=m.withColumn('first_stop',F.min(F.when((F.col('row_event')>0)|(F.col('exit_type')=='other'),
                                           F.col('month_index'))).over(w))
    m=m.filter(F.col('first_stop').isNull()|(F.col('month_index')<=F.col('first_stop')))
    eligible=m.groupBy('loan_id').agg(F.min('month_index').alias('first'),
                  F.max('month_index').alias('last'),F.count('*').alias('n'))
    eligible=eligible.filter((F.col('first')==1)&(F.col('last')==F.col('n'))).select('loan_id')
    m=m.join(eligible,'loan_id','inner')
    final=m.withColumn('rn',F.row_number().over(w.orderBy(F.col('month_index').desc()))).filter('rn=1')
    labels=o.join(final.select('loan_id',F.col('month_index').cast('double').alias('duration'),
                              F.col('row_event').alias('event_type')),'loan_id','inner')
    labels=labels.na.drop().withColumn('event',(F.col('event_type')==1).cast('double')).cache()
    out=a.output.rstrip('/')
    labels.write.mode('overwrite').partitionBy('event_type').parquet(out+'/labels')
    o.join(labels.select('loan_id'),'loan_id','left_anti').write.mode('overwrite').parquet(out+'/excluded_loans')
    # Serializable per-vintage counts rather than collecting the full panel.
    labels.withColumn('vintage',F.date_format('cohort_date','yyyy')).groupBy('vintage','event_type').count()\
        .write.mode('overwrite').parquet(out+'/vintage_counts')
    train=labels.filter(F.col('cohort_date')<=F.lit(a.train_end))
    valid=labels.filter((F.col('cohort_date')>F.lit(a.train_end))&(F.col('cohort_date')<=F.lit(a.valid_end)))
    test=labels.filter(F.col('cohort_date')>F.lit(a.valid_end))
    counts=[x.count() for x in [train,valid,test]]
    if min(counts)<10 or train.filter('event = 1').count()<5:raise ValueError('Insufficient split/event counts')
    features=['fico','dti','ltv','interest_rate','loan_amount']
    pipeline=Pipeline(stages=[VectorAssembler(inputCols=features,outputCol='raw_features'),
                   StandardScaler(inputCol='raw_features',outputCol='features',withMean=True,withStd=True),
                   AFTSurvivalRegression(labelCol='duration',censorCol='event',featuresCol='features',maxIter=100)])
    model=pipeline.fit(train);model.write().overwrite().save(out+'/spark_aft_model')
    model.transform(valid).write.mode('overwrite').parquet(out+'/validation_predictions')
    model.transform(test).write.mode('overwrite').parquet(out+'/test_predictions')
    metadata={'input_mode':'Freddie Standard raw' if a.freddie else 'canonical CSV',
              'included':labels.count(),'split_counts':counts,'seconds':time.perf_counter()-started,
              'spark_version':spark.version,'spark_master':spark.sparkContext.master,
              'target':'first observed D90+','censorCol_semantics':'1=event; 0=censored',
              'estimand':'cause-specific Weibull AFT, payoff censored',
              'limitations':['Restricted to month-1 continuous histories','No IPCW or competing-risk prediction model',
                             'Cohort holdout is retrospective; not an as-of deployment simulation']}
    print(json.dumps(metadata,indent=2))
    # Local metadata companion; use a local output path for this entry point.
    Path(out).mkdir(parents=True,exist_ok=True)
    Path(out,'spark_run.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    spark.stop()

if __name__=='__main__':main()
