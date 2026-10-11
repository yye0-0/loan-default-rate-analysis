import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from panel import build_labels,cumulative_incidence
from features import build_features
from survival import fit_weibull,concordance

class CoreTests(unittest.TestCase):
    def test_event_payoff_gap(self):
        orig=pd.DataFrame({'loan_id':['A','B','C','D','E'],'cohort_date':['2020-01-01']*5,
                           'fico':[700]*5,'dti':[20]*5,'ltv':[80]*5,
                           'interest_rate':[5]*5,'loan_amount':[10000]*5})
        m=pd.DataFrame([('A',1,0,'active'),('A',2,3,'active'),('A',3,0,'active'),
                        ('B',1,0,'active'),('B',2,0,'payoff'),('B',3,3,'active'),
                        ('C',1,0,'active'),('C',3,0,'active'),('D',1,0,'active')],
                       columns=['loan_id','month_index','delinquency_months','exit_type'])
        labels,excluded=build_labels(orig,m);labels=labels.set_index('loan_id')
        self.assertEqual((labels.loc['A','duration'],labels.loc['A','event_type']),(2,1))
        self.assertEqual((labels.loc['B','duration'],labels.loc['B','event_type']),(2,2))
        self.assertEqual(labels.loc['D','event_type'],0)
        self.assertEqual(set(excluded.loan_id),{'C','E'})
        cif=cumulative_incidence(labels)
        self.assertAlmostEqual(cif.iloc[-1].d90_cumulative_incidence,.5)

    def test_no_target_in_features(self):
        raw=pd.DataFrame({'loanAmnt':[1000],'term':[3],'interestRate':[0],'annualIncome':[12000],
              'dti':[20],'ficoRangeLow':[700],'ficoRangeHigh':[710],'issueDate':['2020-01-01'],
              'earliesCreditLine':['2010-01-01'],'grade':['A'],'isDefault':[0]})
        a=build_features(raw);raw.isDefault=1;b=build_features(raw)
        pd.testing.assert_frame_equal(a,b)
        self.assertAlmostEqual(a.monthly_payment.iloc[0],1000/36)

    def test_weibull_direction(self):
        r=np.random.default_rng(5);X=r.normal(size=(600,1))
        actual=np.exp(3+.7*X[:,0])*r.weibull(1.6,600)
        censor=np.full(600,40.);duration=np.minimum(actual,censor);event=actual<=censor
        theta=fit_weibull(X,duration,event)
        self.assertGreater(theta[1],.4);self.assertLess(theta[1],1.0)
        self.assertAlmostEqual(concordance(np.array([1,2,3]),np.ones(3),np.array([1,2,3])),1.)

if __name__=='__main__':unittest.main()
