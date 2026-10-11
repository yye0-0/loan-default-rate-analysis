"""Independent pandas oracle. Time zero is the FIRST observed current month.

Events: 0 censor, 1 D90, 2 payoff, 3 credit-stress exit, 4 other exit.
Terminal events precede end-of-month censoring; D90 takes same-month priority.
"""
import numpy as np
import pandas as pd

FEATURES=['fico','dti','ltv','interest_rate','loan_amount','entry_age_months']
CREDIT_CODES={'02','03','09'}
def ordinal(date):
    x=pd.Timestamp(date);return x.year*12+x.month
def kind(dpd,code):
    try:
        if known(dpd) and float(dpd)>=3:return 1
    except (ValueError,TypeError):pass
    code=str(code).strip().zfill(2) if str(code).strip() else ''
    if code=='01':return 2
    if str(dpd).strip()=='RA' or code in CREDIT_CODES:return 3
    if code not in {'','00','nan','None'}:return 4
    return 0
def current(dpd):
    try:return float(dpd)==0
    except (ValueError,TypeError):return False
def known(dpd):
    try:return 0<=float(dpd)<=99 and float(dpd).is_integer()
    except (ValueError,TypeError):return False
def validate(orig,monthly):
    for frame,cols in [(orig,['loan_id','first_payment_month',*FEATURES[:-1]]),
                       (monthly,['loan_id','report_month','dpd','zero_code','modification'])]:
        if set(cols)-set(frame):raise ValueError('Missing columns')
        if frame.loan_id.isna().any():raise ValueError('Missing loan ID')
    if orig.loan_id.duplicated().any() or monthly.duplicated(['loan_id','report_month']).any():
        raise ValueError('Duplicate loan / loan-month')
    for col,frame in [('first_payment_month',orig),('report_month',monthly)]:
        dates=pd.to_datetime(frame[col],errors='raise')
        if dates.isna().any() or (dates.dt.day!=1).any():raise ValueError('Use month-start labels')
    if not monthly.loan_id.isin(orig.loan_id).all():raise ValueError('Unmatched monthly IDs')
def label_asof(orig,monthly,cutoff):
    validate(orig,monthly);rows=[];audit=[]
    m=monthly[pd.to_datetime(monthly.report_month)<=pd.Timestamp(cutoff)].copy()
    groups={i:g.sort_values('report_month') for i,g in m.groupby('loan_id')}
    for o in orig.to_dict('records'):
        g=groups.get(o['loan_id']);reason='no_history_asof'
        if g is None:audit.append({'loan_id':o['loan_id'],'reason':reason});continue
        records=g.to_dict('records');first=records[0];origin=ordinal(first['report_month'])
        if not current(first['dpd']) or kind(first['dpd'],first['zero_code']) or str(first['modification']) in {'Y','P'}:
            audit.append({'loan_id':o['loan_id'],'reason':'not_current_at_entry'});continue
        duration=0;event=0;reason='administrative';previous=origin
        for r in records[1:]:
            t=ordinal(r['report_month'])-origin
            if ordinal(r['report_month'])!=previous+1:reason='gap';break
            event=kind(r['dpd'],r['zero_code'])
            if event:duration=t;reason='event';break
            if not known(r['dpd']) or str(r['modification']) in {'Y','P'}:
                event=0;reason='unknown_or_modification';break
            duration=t;previous=ordinal(r['report_month'])
        if duration<1:
            audit.append({'loan_id':o['loan_id'],'reason':'no_followup_interval'});continue
        rows.append({**o,'snapshot_month':first['report_month'],
          'entry_age_months':max(0,origin-ordinal(o['first_payment_month'])),
          'duration':duration,'event_type':event,'d90_event':int(event==1),'stop_reason':reason})
        audit.append({'loan_id':o['loan_id'],'reason':reason})
    return pd.DataFrame(rows),pd.DataFrame(audit)
def cumulative_incidence(labels):
    s=1.;cif=np.zeros(4);rows=[]
    for t in sorted(labels.duration.unique()):
        risk=(labels.duration>=t).sum();k=labels.loc[labels.duration==t,'event_type']
        d=np.array([(k==i).sum() for i in range(1,5)])
        cif+=s*d/risk;s*=1-d.sum()/risk
        rows.append({'month':int(t),'at_risk':int(risk),'event_free':s,
                     **{f'cif_{i+1}':float(v) for i,v in enumerate(cif)}})
    return pd.DataFrame(rows)
