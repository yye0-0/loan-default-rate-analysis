"""Age-standardized observed calendar D90 hazard; aggregate in Spark."""
import argparse,json
from pathlib import Path
import pandas as pd
from spark_pipeline import make_spark,dataset_parts,label_asof
def main():
    from pyspark.sql import functions as F
    p=argparse.ArgumentParser();p.add_argument('--spark-dir',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--master',default='local[2]');a=p.parse_args();config=json.loads(Path(a.config).read_text())
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);s=make_spark(a.master)
    o=s.read.parquet(*dataset_parts(Path(a.spark_dir)/'origination'));m=s.read.parquet(*dataset_parts(Path(a.spark_dir)/'monthly'))
    cutoff=config['phases']['validation']['asof'];labels,_=label_asof(o,m,cutoff)
    x=labels.withColumn('t',F.explode(F.sequence(F.lit(1),F.col('duration'))))
    x=x.withColumn('calendar_month',F.add_months('snapshot_month',F.col('t'))).withColumn('age',F.col('entry_age_months')+F.col('t'))
    x=x.withColumn('age_band',F.when(F.col('age')<=6,'1-6').when(F.col('age')<=12,'7-12').when(F.col('age')<=24,'13-24').otherwise('25+'))
    x=x.withColumn('d90',((F.col('event_type')==1)&(F.col('t')==F.col('duration'))).cast('int'))
    agg=x.groupBy('calendar_month','age_band').agg(F.count('*').alias('at_risk'),F.sum('d90').alias('events')).toPandas()
    weights=x.filter(F.col('calendar_month')<=F.lit(config['phases']['train']['asof'])).groupBy('age_band').count().toPandas()
    weights=dict(zip(weights.age_band,weights['count']/weights['count'].sum()));rows=[]
    for date,g in agg.groupby('calendar_month'):
        g=g.set_index('age_band');supported=set(g.index)==set(weights) and int(g.at_risk.sum())>=100 and int(g.at_risk.min())>=10
        rate=sum(weights[b]*g.loc[b,'events']/g.loc[b,'at_risk'] for b in weights) if supported else None
        rows.append({'date':str(pd.Timestamp(date).date()),'events':int(g.events.sum()),'at_risk':int(g.at_risk.sum()),
           'raw_rate':float(g.events.sum()/g.at_risk.sum()),'rate':rate,'supported':supported})
    pd.DataFrame(rows).sort_values('date').to_csv(out/'calendar_hazard.csv',index=False)
    agg.to_csv(out/'age_band_counts.csv',index=False)
    (out/'calendar_report.json').write_text(json.dumps({'asof':cutoff,'age_weights':weights,'weight_fit_cutoff':config['phases']['train']['asof'],
      'target':'Observed conditional calendar D90 hazard; no portfolio/default-rate identity claim',
      'support':'at risk >=100, all four age bands each >=10','months':len(rows)},indent=2));s.stop()
if __name__=='__main__':main()
