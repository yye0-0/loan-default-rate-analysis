"""Real Spark ETL. Window labels, bounded export, immutable part manifest."""
import argparse,hashlib,json,time,shutil
from pathlib import Path
from panel import FEATURES

def make_spark(master='local[2]'):
    from pyspark.sql import SparkSession
    s=(SparkSession.builder.master(master).appName('LoanDefaultTimeRisk')
       .config('spark.ui.enabled','false').config('spark.sql.shuffle.partitions','4')
       .config('spark.driver.memory','1g').getOrCreate());s.sparkContext.setLogLevel('ERROR');return s
def read_inputs(spark,orig,monthly,version='canonical'):
    from pyspark.sql import functions as F
    if version=='canonical':
        o=spark.read.option('header',True).option('mode','FAILFAST').csv(str(orig))
        m=spark.read.option('header',True).option('mode','FAILFAST').csv(str(monthly))
        for frame,fields in [(o,['loan_id','first_payment_month',*FEATURES[:-1]]),
                             (m,['loan_id','report_month','dpd','zero_code','modification'])]:
            if set(fields)-set(frame.columns):raise ValueError('Missing canonical columns')
        o=o.withColumn('first_payment_month',F.to_date('first_payment_month'))
        m=m.withColumn('report_month',F.to_date('report_month'))
    else:
        def raw(path,width):
            t=spark.read.text(str(path)).withColumn('a',F.split('value',r'\|',-1))
            bad=F.size('a')!=width if version=='july2026' else F.size('a')<width
            if t.filter(bad).limit(1).count():raise ValueError('Unexpected vendor field count')
            return t
        ro=raw(orig,31 if version=='july2026' else 20)
        rm=raw(monthly,35 if version=='july2026' else 9)
        at=lambda k:F.trim(F.col('a')[k])
        o=ro.select(at(19).alias('loan_id'),F.to_date(at(1),'yyyyMM').alias('first_payment_month'),
           at(0).alias('fico'),at(9).alias('dti'),at(11).alias('ltv'),at(12).alias('interest_rate'),at(10).alias('loan_amount'))
        m=rm.select(at(0).alias('loan_id'),F.to_date(at(1),'yyyyMM').alias('report_month'),
           at(3).alias('dpd'),at(8).alias('zero_code'),at(7).alias('modification'))
    o=o.withColumn('loan_id',F.trim('loan_id'));m=m.withColumn('loan_id',F.trim('loan_id'))
    for col in FEATURES[:-1]:o=o.withColumn(col,F.col(col).cast('double'))
    for col in ['dpd','zero_code','modification']:m=m.withColumn(col,F.trim(F.coalesce(F.col(col),F.lit(''))))
    bounds={'fico':(300,850),'dti':(0,65),'ltv':(1,998),'interest_rate':(.001,100),'loan_amount':(1,1e10)}
    for col,(lo,hi) in bounds.items():
        o=o.withColumn(col,F.when(F.col(col).between(lo,hi),F.col(col)))
    validate(o,m);return o,m
def validate(o,m):
    from pyspark.sql import functions as F
    for frame,col in [(o,'first_payment_month'),(m,'report_month')]:
        if frame.filter(F.col('loan_id').isNull()|(F.col('loan_id')=='')|F.col(col).isNull()|(F.dayofmonth(col)!=1)).limit(1).count():
            raise ValueError('Missing IDs / invalid month labels')
    if o.groupBy('loan_id').count().filter('count>1').limit(1).count():raise ValueError('Duplicate loan')
    if m.groupBy('loan_id','report_month').count().filter('count>1').limit(1).count():raise ValueError('Duplicate loan-month')
    if m.join(o.select('loan_id'),'loan_id','left_anti').limit(1).count():raise ValueError('Unmatched monthly IDs')
def label_asof(o,m,cutoff):
    from pyspark.sql import functions as F,Window
    w=Window.partitionBy('loan_id').orderBy('report_month')
    allw=w.rowsBetween(Window.unboundedPreceding,Window.unboundedFollowing)
    x=m.filter(F.col('report_month')<=F.lit(cutoff))
    x=x.withColumn('num',F.col('dpd').cast('double'))
    code=F.lpad('zero_code',2,'0')
    valid_numeric=F.col('num').between(0,99)&(F.col('num')==F.floor('num'))
    x=x.withColumn('row_kind',F.when(valid_numeric&(F.col('num')>=3),1).when(code=='01',2)
         .when((F.col('dpd')=='RA')|code.isin('02','03','09'),3)
         .when(~code.isin('','00'),4).otherwise(0))
    x=x.withColumn('ordinal',F.year('report_month')*12+F.month('report_month'))
    x=x.withColumn('snapshot_month',F.first('report_month').over(allw))
    x=x.withColumn('origin',F.first('ordinal').over(allw))
    x=x.withColumn('eligible',F.first((F.col('num')==0)&(F.col('row_kind')==0)&(~F.col('modification').isin('Y','P'))).over(allw))
    x=x.withColumn('offset',F.col('ordinal')-F.col('origin'))
    x=x.withColumn('prev_ord',F.lag('ordinal').over(w)).withColumn('prev_offset',F.lag('offset').over(w))
    gap=(F.col('offset')>0)&(F.col('ordinal')!=F.col('prev_ord')+1)
    unknown=(F.col('offset')>0)&(F.col('row_kind')==0)&(
        F.col('num').isNull()|(~valid_numeric)|F.col('modification').isin('Y','P'))
    terminal=(F.col('offset')>0)&(F.col('row_kind')>0)
    x=x.withColumn('stop',F.when(gap|unknown,F.struct(F.col('prev_offset').alias('duration'),F.lit(0).alias('event_type'),
          F.when(gap,'gap').otherwise('unknown_or_modification').alias('stop_reason')))
          .when(terminal,F.struct(F.col('offset').alias('duration'),F.col('row_kind').alias('event_type'),F.lit('event').alias('stop_reason'))))
    g=x.groupBy('loan_id').agg(F.first('snapshot_month').alias('snapshot_month'),F.first('eligible').alias('eligible'),
        F.max('offset').alias('last'),F.min('stop').alias('stop'))
    g=g.withColumn('duration',F.coalesce(F.col('stop.duration'),F.col('last')))
    g=g.withColumn('event_type',F.coalesce(F.col('stop.event_type'),F.lit(0)))
    g=g.withColumn('stop_reason',F.coalesce(F.col('stop.stop_reason'),F.lit('administrative')))
    audits=o.select('loan_id').join(g,'loan_id','left').select('loan_id',
        F.when(F.col('snapshot_month').isNull(),'no_history_asof').when(~F.coalesce('eligible',F.lit(False)),'not_current_at_entry')
         .when(F.col('duration')<1,'no_followup_interval').otherwise(F.col('stop_reason')).alias('reason'))
    labels=o.join(g.filter(F.coalesce('eligible',F.lit(False))&(F.col('duration')>=1)),'loan_id')
    labels=labels.withColumn('entry_age_months',F.greatest(F.lit(0),
        F.year('snapshot_month')*12+F.month('snapshot_month')-F.year('first_payment_month')*12-F.month('first_payment_month')))
    labels=labels.withColumn('d90_event',(F.col('event_type')==1).cast('int'))
    return labels.select('loan_id','first_payment_month',*FEATURES,'snapshot_month','duration','event_type','d90_event','stop_reason'),audits
def person_period(labels,horizon=24):
    from pyspark.sql import functions as F
    return labels.withColumn('t',F.explode(F.sequence(F.lit(1),F.least(F.col('duration'),F.lit(horizon)))))\
      .withColumn('outcome',F.when(F.col('t')==F.col('duration'),F.col('event_type')).otherwise(0))
def write_parquet(frame,path):
    path=Path(path)
    if path.exists():shutil.rmtree(path)
    frame.write.mode('overwrite').parquet(str(path))
    parts=sorted(p.name for p in path.glob('part-*.parquet'))
    if not parts:raise RuntimeError('No committed parts')
    (path/'_parts_manifest.json').write_text(json.dumps(parts))
def dataset_parts(path):
    p=Path(path);manifest=p/'_parts_manifest.json'
    names=json.loads(manifest.read_text()) if manifest.exists() else sorted(q.name for q in p.glob('part-*.parquet'))
    if not names or len(names)!=len(set(names)) or any(Path(n).name!=n or not (p/n).is_file() for n in names):
        raise ValueError('Invalid or empty dataset manifest')
    return [str(p/n) for n in names]
def read_local(path):
    import pandas as pd
    return pd.concat([pd.read_parquet(p) for p in dataset_parts(path)],ignore_index=True)
def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
def main():
    from pyspark.sql import functions as F
    p=argparse.ArgumentParser();p.add_argument('--orig',required=True);p.add_argument('--monthly',required=True)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--master',default='local[2]')
    p.add_argument('--freddie-version',choices=['canonical','legacy','july2026'],default='canonical');a=p.parse_args()
    config=json.loads(Path(a.config).read_text())
    from risk_models import validate_config
    validate_config(config);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    s=make_spark(a.master);started=time.perf_counter();o,m=read_inputs(s,a.orig,a.monthly,a.freddie_version);o.cache();m.cache()
    meta={'engine':'Apache Spark','spark_version':s.version,'master':s.sparkContext.master,'data_kind':config['data_kind'],
       'orig_rows':o.count(),'monthly_rows':m.count(),'input_version':a.freddie_version,'phases':{},
       'input_sha256':{n:sha256(v) for n,v in [('orig',a.orig),('monthly',a.monthly),('config',a.config)]}}
    write_parquet(o,out/'origination');write_parquet(m,out/'monthly')
    for phase,c in config['phases'].items():
        labels,audit=label_asof(o,m,c['asof']);labels=labels.filter(F.col('snapshot_month').between(c['start'],c['end'])).cache()
        write_parquet(labels,out/f'labels_{phase}');write_parquet(audit,out/f'audit_{phase}')
        counts={str(r.event_type):r['count'] for r in labels.groupBy('event_type').count().collect()}
        meta['phases'][phase]={'loans':labels.count(),'events':counts,
          'included_stop_reasons':{r.stop_reason:r['count'] for r in labels.groupBy('stop_reason').count().collect()},
          'all_input_audit_reasons':{r.reason:r['count'] for r in audit.groupBy('reason').count().collect()},**c}
        # Bounded local models: no collection of the full raw monthly table.
        bounded=labels.orderBy('loan_id').limit(6000).cache()
        write_parquet(bounded,out/f'model_labels_{phase}');write_parquet(person_period(bounded),out/f'period_{phase}')
        labels.unpersist();bounded.unpersist()
    meta['quality']={'missing_features':{c:o.filter(F.col(c).isNull()).count() for c in FEATURES[:-1]},
         'modification_rows':m.filter(F.col('modification').isin('Y','P')).count(),
         'transfer_15_16_rows':m.filter(F.col('zero_code').isin('15','16')).count(),
         'dpd_counts':{r.dpd:r['count'] for r in m.groupBy('dpd').count().collect()},
         'same_month_numeric_d90_and_exit_rows':m.filter(F.col('dpd').cast('double').between(3,99)&(~F.col('zero_code').isin('','00'))).count()}
    meta['elapsed_seconds']=time.perf_counter()-started
    (out/'spark_run.json').write_text(json.dumps(meta,indent=2));s.stop();print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
