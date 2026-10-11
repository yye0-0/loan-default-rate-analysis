import sys,unittest,tempfile,shutil,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd
from panel import label_asof,kind,cumulative_incidence,FEATURES
from metrics import CensorKM,horizon_target
from risk_models import validate_config,prep,predict_cif
from make_demo_v2 import CONFIG,generate
from features import build_features

def fixture():
    ids=list('ABCDEFGHIJKL')
    o=pd.DataFrame({'loan_id':ids,'first_payment_month':['2019-01-01']*len(ids),
      'fico':[700.]*len(ids),'dti':[25.]*len(ids),'ltv':[80.]*len(ids),
      'interest_rate':[5.]*len(ids),'loan_amount':[200000.]*len(ids)})
    rows=[]
    def add(lid,month,dpd='0',code='',mod='N'):rows.append((lid,f'2019-{month:02d}-01',dpd,code,mod))
    add('A',3);add('A',4,'3');add('A',5,'RA','09','Y')
    add('B',1);add('B',2,'0','01');add('B',3,'3')
    add('C',1);add('C',2);add('C',4)
    add('D',1);add('D',2);add('D',3,'XX')
    add('E',1);add('E',2,'0','15')
    add('F',1);add('F',2,'0','16')
    add('G',1);add('G',2,'RA','09')
    add('H',1,'1');add('H',2)
    add('I',1,'0','','Y');add('I',2)
    add('K',1);add('K',2,'3','01')
    add('L',1);add('L',2);add('L',3,'3');add('L',4,'RA','09','Y')
    return o,pd.DataFrame(rows,columns=['loan_id','report_month','dpd','zero_code','modification'])
class LabelTests(unittest.TestCase):
    def test_first_events_and_delayed_entry(self):
        o,m=fixture();l,a=label_asof(o,m,'2019-06-01');l=l.set_index('loan_id')
        self.assertEqual((l.loc['A','duration'],l.loc['A','event_type'],l.loc['A','entry_age_months']),(1,1,2))
        self.assertEqual(l.loc['B','event_type'],2)
        self.assertEqual(set(a[a.reason.isin(['no_history_asof','not_current_at_entry'])].loan_id),{'H','I','J'})
    def test_gap_keeps_only_known_months(self):
        o,m=fixture();l,_=label_asof(o,m,'2019-06-01');r=l.set_index('loan_id').loc['C']
        self.assertEqual((r.duration,r.event_type,r.stop_reason),(1,0,'gap'))
    def test_unknown_is_not_current(self):
        o,m=fixture();l,_=label_asof(o,m,'2019-06-01');r=l.set_index('loan_id').loc['D']
        self.assertEqual((r.duration,r.event_type),(1,0))
    def test_invalid_numeric_status_is_unknown(self):
        o,m=fixture()
        for value in ['999','3.5','Infinity']:
            m.loc[(m.loan_id=='D')&(m.report_month=='2019-03-01'),'dpd']=value
            l,_=label_asof(o,m,'2019-06-01');r=l.set_index('loan_id').loc['D']
            self.assertEqual((r.duration,r.event_type),(1,0))
            self.assertEqual(kind(value,''),0)
    def test_transfers_not_credit_events(self):
        self.assertEqual(kind('0','15'),4);self.assertEqual(kind('0','16'),4);self.assertEqual(kind('RA','09'),3)
    def test_same_month_priority(self):self.assertEqual(kind('3','01'),1)
    def test_future_rows_do_not_change_past(self):
        o,m=fixture();a,_=label_asof(o,m,'2019-02-01');b,_=label_asof(o,m[pd.to_datetime(m.report_month)<=pd.Timestamp('2019-02-01')],'2019-02-01')
        pd.testing.assert_frame_equal(a,b)
    def test_later_ra_modification_preserves_d90(self):
        o,m=fixture();a,_=label_asof(o,m,'2019-06-01');b,_=label_asof(o,m[~m.modification.eq('Y')],'2019-06-01')
        pd.testing.assert_frame_equal(a[a.loan_id=='A'].reset_index(drop=True),b[b.loan_id=='A'].reset_index(drop=True))
    def test_duplicate_rejected(self):
        o,m=fixture()
        with self.assertRaises(ValueError):label_asof(o,pd.concat([m,m.iloc[[0]]]),'2019-06-01')
    def test_month_start_rejected(self):
        o,m=fixture();m.loc[0,'report_month']='2019-03-02'
        with self.assertRaises(ValueError):label_asof(o,m,'2019-06-01')
    def test_empirical_mass_conservation(self):
        o,m=fixture();l,_=label_asof(o,m,'2019-06-01');f=cumulative_incidence(l)
        np.testing.assert_allclose(f[['event_free','cif_1','cif_2','cif_3','cif_4']].sum(axis=1),1)
    def test_censor_after_month_and_terminal_ties(self):
        f=pd.DataFrame({'duration':[1,1,2,2],'event_type':[1,0,2,0]});y,w,s=horizon_target(f,2,CensorKM().fit(f))
        np.testing.assert_allclose(w,[1,0,1.5,1.5]);self.assertEqual(y.tolist(),[1,0,0,0])
    def test_future_training_outcomes_rejected(self):
        c=json.loads(json.dumps(CONFIG));c['phases']['train']['asof']='2019-12-01'
        with self.assertRaises(ValueError):validate_config(c)
    def test_target_does_not_change_features(self):
        raw=pd.DataFrame({'loanAmnt':[1000],'term':[3],'interestRate':[0],'annualIncome':[12000],
          'dti':[20],'ficoRangeLow':[700],'ficoRangeHigh':[710],'issueDate':['2020-01-01'],
          'earliesCreditLine':['2010-01-01'],'grade':['A'],'isDefault':[0]})
        a=build_features(raw);raw.isDefault=1;pd.testing.assert_frame_equal(a,build_features(raw))
    def test_train_only_imputation(self):
        p=prep(['fico']);p.fit(pd.DataFrame({'fico':[600.,800.,np.nan]}))
        med=p.named_transformers_['numeric'].named_steps['impute'].statistics_.copy()
        p.transform(pd.DataFrame({'fico':[9000.,np.nan]}));np.testing.assert_equal(med,[700.])
    def test_joint_curve_mass_and_monotonicity_after_calibration(self):
        class Fixed:
            classes_=np.arange(5)
            def predict_proba(self,x):return np.tile([.95,.02,.02,.005,.005],(len(x),1))
        frame=pd.DataFrame({c:[700.] for c in FEATURES});frame.loan_amount=200000.
        out=predict_cif(Fixed(),frame,theta=np.array([.3,.2,-.1,.1,.4]))
        for v in out.values():self.assertAlmostEqual(v.sum(),1)
        self.assertTrue(np.all(out[24][:,1:]>=out[12][:,1:]));self.assertLess(out[24][0,0],out[12][0,0])
class SparkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from spark_pipeline import make_spark
        cls.s=make_spark()
    @classmethod
    def tearDownClass(cls):cls.s.stop()
    def test_spark_matches_independent_oracle_two_cutoffs(self):
        from spark_pipeline import label_asof as spark_labels
        o,m=fixture();so=self.s.createDataFrame(o);sm=self.s.createDataFrame(m)
        from pyspark.sql import functions as F
        so=so.withColumn('first_payment_month',F.to_date('first_payment_month'));sm=sm.withColumn('report_month',F.to_date('report_month'))
        cols=['loan_id','duration','event_type','entry_age_months','stop_reason']
        for cutoff in ['2019-02-01','2019-06-01']:
            expected,_=label_asof(o,m,cutoff);actual,_=spark_labels(so,sm,cutoff)
            pd.testing.assert_frame_equal(expected[cols].sort_values('loan_id').reset_index(drop=True),
             actual.toPandas()[cols].sort_values('loan_id').reset_index(drop=True),check_dtype=False)
        sm=sm.withColumn('dpd',F.when((F.col('loan_id')=='D')&(F.col('report_month')==F.lit('2019-03-01')),F.lit('999')).otherwise(F.col('dpd')))
        actual,_=spark_labels(so,sm,'2019-06-01');r=actual.filter("loan_id='D'").first()
        self.assertEqual((r.duration,r.event_type),(1,0))
    def test_manifest_ignores_stale_parts(self):
        from spark_pipeline import write_parquet,read_local
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'parts';write_parquet(self.s.range(10),path)
            backup=Path(tmp)/'old.parquet';shutil.copy2(next(path.glob('part-*.parquet')),backup)
            write_parquet(self.s.range(3),path);shutil.copy2(backup,path/'part-stale.parquet')
            self.assertEqual(len(read_local(path)),3)
    def test_july2026_adapter_and_width_guard(self):
        from spark_pipeline import read_inputs
        with tempfile.TemporaryDirectory() as tmp:
            generate(tmp,20);p=Path(tmp);o,m=read_inputs(self.s,p/'freddie_orig_synthetic.txt',p/'freddie_perf_synthetic.txt','july2026')
            self.assertEqual(o.count(),20);self.assertGreater(m.count(),20)
            (p/'bad.txt').write_text('|'.join(['']*32)+'\n')
            with self.assertRaises(ValueError):read_inputs(self.s,p/'freddie_orig_synthetic.txt',p/'bad.txt','july2026')
if __name__=='__main__':unittest.main()
