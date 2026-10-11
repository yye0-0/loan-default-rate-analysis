"""Reconstructed ideas, not the lost 2024 source code. No outcome features."""
import numpy as np
import pandas as pd

NUMERIC = ['loanAmnt','term','interestRate','annualIncome','dti','ficoRangeMean',
           'revolUtil','credit_history_length','loan_to_income','monthly_payment',
           'payment_to_income_ratio','total_loan_cost','risk_composite','capm_grade_proxy']
CATEGORICAL = ['grade','homeOwnership','verificationStatus','purpose','applicationType']

def build_features(raw):
    required = ['loanAmnt','term','interestRate','annualIncome','dti',
                'ficoRangeLow','ficoRangeHigh','issueDate','grade']
    missing = set(required)-set(raw.columns)
    if missing:
        raise ValueError(f'Missing fields: {sorted(missing)}')
    # Explicit whitelist prevents arbitrary source fields/target entering training.
    f = pd.DataFrame(index=raw.index)
    for c in ['loanAmnt','interestRate','annualIncome','dti','ficoRangeLow','ficoRangeHigh']:
        f[c] = pd.to_numeric(raw[c],errors='coerce')
    # Original Tianchi term is in years. This adapter requires years, not months.
    f['term'] = pd.to_numeric(raw['term'],errors='coerce')
    if not f['term'].dropna().isin([3,5]).all():
        raise ValueError('term must use years (3 or 5). Convert 36/60 months explicitly.')
    f['ficoRangeMean'] = (f.pop('ficoRangeLow')+f.pop('ficoRangeHigh'))/2
    f['revolUtil'] = pd.to_numeric(raw.get('revolUtil',pd.Series(np.nan,index=raw.index)),errors='coerce')
    for c in CATEGORICAL:
        s = raw[c] if c in raw else pd.Series('UNKNOWN',index=raw.index)
        f[c] = s.fillna('UNKNOWN').astype(str)
    issue = pd.to_datetime(raw['issueDate'],errors='raise')
    # earliest credit line may be a formatted date in the original CSV: normalize first.
    if 'earliesCreditLine' in raw:
        early = pd.to_datetime(raw['earliesCreditLine'],errors='coerce')
        f['credit_history_length'] = (issue-early).dt.days/365.25
        f.loc[f.credit_history_length<0,'credit_history_length'] = np.nan
    else:
        f['credit_history_length'] = np.nan
    income = f.annualIncome.where(f.annualIncome>0)
    f['loan_to_income'] = f.loanAmnt/income
    r = f.interestRate/1200.0
    n = f.term*12
    denominator = -np.expm1(-n*np.log1p(r))
    f['monthly_payment'] = np.where(r==0,f.loanAmnt/n,f.loanAmnt*r/denominator)
    f['payment_to_income_ratio'] = f.monthly_payment/(income/12)
    f['total_loan_cost'] = f.monthly_payment*n-f.loanAmnt
    # New illustrative composite; NOT Altman Z-score and not the unavailable old weights.
    f['risk_composite'] = ((f.ficoRangeMean-300)/550 - f.loan_to_income
                           - f.dti/100 - f.revolUtil/100)
    # Fixed grade mapping is a heuristic; no estimated covariance beta or market CAPM fit.
    beta = f.grade.map(dict(zip('ABCDEFG',np.linspace(.5,3.9,7))))
    f['capm_grade_proxy'] = .03+beta*(.10-.03)
    return f[NUMERIC+CATEGORICAL].replace([np.inf,-np.inf],np.nan)

def temporal_split(raw, train_end, valid_end):
    date = pd.to_datetime(raw['issueDate'],errors='raise')
    a,b = pd.Timestamp(train_end),pd.Timestamp(valid_end)
    if a>=b:
        raise ValueError('train_end must precede valid_end')
    masks = [date<=a,(date>a)&(date<=b),date>b]
    if any(m.sum()==0 for m in masks):
        raise ValueError('Empty split: select boundaries appropriate to the dataset')
    return masks
