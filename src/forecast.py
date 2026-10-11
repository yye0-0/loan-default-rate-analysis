"""Two rolling validation windows + untouched 12-month calendar holdout."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit,logit

def forecast(train,horizon=12):
    from statsmodels.tsa.arima.model import ARIMA
    import torch
    from torch import nn
    torch.manual_seed(42);torch.set_num_threads(2)
    y=logit(np.clip(train,1e-5,1-1e-5));fit=ARIMA(y,order=(1,0,1)).fit()
    base=np.asarray(fit.forecast(horizon));resid=np.asarray(fit.resid)[3:];mu=resid.mean();sd=resid.std()+1e-6;z=(resid-mu)/sd
    lookback=6;xs=np.array([z[i-lookback:i] for i in range(lookback,len(z))])[:,:,None];ys=z[lookback:]
    class ResidualLSTM(nn.Module):
        def __init__(self):super().__init__();self.lstm=nn.LSTM(1,8,batch_first=True);self.head=nn.Linear(8,1)
        def forward(self,x):h,_=self.lstm(x);return self.head(h[:,-1,:]).squeeze(-1)
    model=ResidualLSTM();opt=torch.optim.Adam(model.parameters(),lr=.01)
    xt=torch.tensor(xs,dtype=torch.float32);yt=torch.tensor(ys,dtype=torch.float32)
    for _ in range(60):
        opt.zero_grad();loss=nn.functional.mse_loss(model(xt),yt);loss.backward();opt.step()
    history=list(z);correction=[];model.eval()
    with torch.no_grad():
        for _ in range(horizon):
            nxt=float(model(torch.tensor(history[-lookback:],dtype=torch.float32).reshape(1,lookback,1)))
            history.append(nxt);correction.append(nxt*sd+mu)
    return {'last_value':np.full(horizon,train[-1]),'seasonal_naive':np.tile(train[-12:],int(np.ceil(horizon/12)))[:horizon],
            'arima':expit(base),'arima_lstm':expit(base+np.asarray(correction))}
def metrics(actual,pred):
    return {k:{'mae':float(np.mean(abs(actual-v))),'rmse':float(np.sqrt(np.mean((actual-v)**2)))} for k,v in pred.items()}
def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    f=pd.read_csv(a.csv).sort_values('date');supported=f.rate.notna()
    if not supported.any():raise ValueError('No supported months')
    first=np.flatnonzero(supported)[0];f=f.iloc[first:].reset_index(drop=True)
    dates=pd.to_datetime(f.date).dt.to_period('M')
    if f.rate.isna().any() or dates.duplicated().any() or len(dates)!=len(pd.period_range(dates.min(),dates.max(),freq='M')):raise ValueError('Unsupported gap within calendar series')
    y=f.rate.to_numpy(float)
    if len(y)<84 or np.any((y<0)|(y>1)):raise ValueError('Need >=84 supported months')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);validation=[]
    for start in [len(y)-36,len(y)-24]:
        pred=forecast(y[:start]);validation.append({'start':f.date.iloc[start],'end':f.date.iloc[start+11],'metrics':metrics(y[start:start+12],pred)})
    average={k:float(np.mean([v['metrics'][k]['mae'] for v in validation])) for k in validation[0]['metrics']}
    selected=min(average,key=average.get)
    freeze={'selected':selected,'validation_mae_mean':average,'selection':'two earlier 12-month windows; fixed ARIMA(1,0,1), LSTM 8 units/60 epochs',
            'holdout_start':f.date.iloc[-12]}
    (out/'frozen_forecast_choice.json').write_text(json.dumps(freeze,indent=2))
    # Only now evaluate the last 12 months.
    pred=forecast(y[:-12]);test=metrics(y[-12:],pred)
    pd.DataFrame({'date':f.date.iloc[-12:].to_numpy(),'actual':y[-12:],**pred}).to_csv(out/'forecast.csv',index=False)
    report={'supported_months':len(f),'trimmed_startup_months':int(first),'validation':validation,'selection':freeze,'test':test,
         'limits':['Observed hazard only; no causal or production forecast claim','Synthetic series tests do not show borrower business value',
                   'No uncertainty bands or tuning search; LSTM advantage not assumed']}
    (out/'forecast_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
