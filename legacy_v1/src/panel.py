"""Canonical monthly panel rules shared with spark_panel.py.

One row per loan-month. month_index=1 is the first full observed risk interval.
This prototype restricts to fully observed histories from month 1, no delayed entry.
event_type: 0 administrative/other exit censoring, 1 first D90+, 2 full payoff.
"""
import numpy as np
import pandas as pd

ORIG = ['loan_id','cohort_date','fico','dti','ltv','interest_rate','loan_amount']
MONTHLY = ['loan_id','month_index','delinquency_months','exit_type']

def validate(orig,monthly):
    for frame,cols in [(orig,ORIG),(monthly,MONTHLY)]:
        missing = set(cols)-set(frame.columns)
        if missing:
            raise ValueError(f'Missing fields: {sorted(missing)}')
        if frame[cols].isna().any().any():
            raise ValueError('Null required fields: normalize or quarantine explicitly')
    if orig.loan_id.duplicated().any() or monthly.duplicated(['loan_id','month_index']).any():
        raise ValueError('Duplicate loan / loan-month')
    if not monthly.loan_id.isin(orig.loan_id).all():
        raise ValueError('Monthly rows without origination records')
    if (monthly.month_index<1).any() or (monthly.month_index%1!=0).any():
        raise ValueError('month_index must be positive integer')
    if (monthly.delinquency_months<0).any():
        raise ValueError('Unknown delinquency sentinel must not become zero')
    if not monthly.exit_type.isin(['active','payoff','other']).all():
        raise ValueError('exit_type must be active, payoff or other')

def build_labels(orig,monthly):
    validate(orig,monthly)
    m = monthly.copy().sort_values(['loan_id','month_index'])
    m['row_event'] = np.where(m.delinquency_months>=3,1,np.where(m.exit_type=='payoff',2,0))
    m['stop'] = (m.row_event>0)|(m.exit_type=='other')
    # Administrative records after a first event/exit do not extend observation time.
    first_stop = m.loc[m.stop].groupby('loan_id').month_index.min()
    m = m[m.month_index<=m.loan_id.map(first_stop).fillna(np.inf)]
    audit = m.groupby('loan_id').month_index.agg(['min','max','count'])
    eligible = (audit['min']==1)&(audit['max']==audit['count'])
    excluded = audit.loc[~eligible].reset_index()
    m = m[m.loan_id.isin(audit.index[eligible])]
    last = m.groupby('loan_id').tail(1)
    result = last[['loan_id','month_index','row_event','exit_type']].rename(
        columns={'month_index':'duration','row_event':'event_type'})
    result['event'] = (result.event_type==1).astype(int)
    result = orig.merge(result,on='loan_id',how='inner',validate='one_to_one')
    # Loans with no panel at all are also explicitly excluded.
    absent = orig.loc[~orig.loan_id.isin(audit.index),['loan_id']].copy()
    excluded = pd.concat([excluded.assign(reason='late_entry_or_gap'),
                          absent.assign(reason='no_monthly_history')],ignore_index=True)
    return result,excluded

def cumulative_incidence(labels):
    """Discrete-time Aalen-Johansen, ties evaluated using the same risk set."""
    survival,cif = 1.,0.
    rows = []
    for t in sorted(labels.duration.unique()):
        at_risk = int((labels.duration>=t).sum())
        events = labels.loc[labels.duration==t,'event_type']
        d1,d2 = int((events==1).sum()),int((events==2).sum())
        cif += survival*d1/at_risk
        survival *= 1-(d1+d2)/at_risk
        rows.append({'month':int(t),'at_risk':at_risk,'d90_events':d1,'payoffs':d2,
                     'd90_cumulative_incidence':cif,'event_free_survival':survival})
    return pd.DataFrame(rows)

def calendar_hazard(orig,monthly):
    """First D90+ events / beginning-of-month at-risk loans. Complete-history subset."""
    labels,_ = build_labels(orig,monthly)
    m = monthly.merge(labels[['loan_id','cohort_date','duration','event_type']],on='loan_id')
    m = m[m.month_index<=m.duration].copy()
    cohort = pd.to_datetime(m.cohort_date).dt.to_period('M')
    # cohort_date is month of risk interval 1, NOT necessarily loan origination.
    m['calendar_month'] = [str(a+int(b)-1) for a,b in zip(cohort,m.month_index)]
    m['new_d90'] = ((m.month_index==m.duration)&(m.event_type==1)).astype(int)
    rate = m.groupby('calendar_month').agg(events=('new_d90','sum'),at_risk=('loan_id','count'))
    rate['rate'] = rate.events/rate.at_risk
    return rate.reset_index().rename(columns={'calendar_month':'date'})
