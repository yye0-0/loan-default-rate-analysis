"""Reconstruct ARIMA + residual LSTM on CALENDAR hazard, not final status by vintage.
Dependency-heavy entry point, not validated in the supplied environment.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    from statsmodels.tsa.arima.model import ARIMA
    import torch
    from torch import nn
    p=argparse.ArgumentParser()
    p.add_argument('--csv',required=True);p.add_argument('--output',required=True)
    p.add_argument('--test-months',type=int,default=12)
    p.add_argument('--lookback',type=int,default=6);p.add_argument('--epochs',type=int,default=80)
    a=p.parse_args()
    df=pd.read_csv(a.csv).sort_values('date')
    idx=pd.to_datetime(df.date).dt.to_period('M')
    if idx.duplicated().any() or len(idx)!=len(pd.period_range(idx.min(),idx.max(),freq='M')):
        raise ValueError('Calendar series must be monthly, unique and contiguous')
    rates=df.rate.to_numpy(float)
    if np.any((rates<0)|(rates>1)) or len(rates)<a.test_months+36:
        raise ValueError('Need >=36 train months plus holdout, rates in [0,1]')
    train,test=rates[:-a.test_months],rates[-a.test_months:]
    # Logit stabilizes probabilities. Fixed ARIMA order avoids holdout tuning.
    y=np.log(np.clip(train,1e-5,1-1e-5)/(1-np.clip(train,1e-5,1-1e-5)))
    fit=ARIMA(y,order=(1,0,1)).fit()
    base=np.asarray(fit.forecast(len(test)))
    # One-step residuals, computed after ARIMA fit on train only.
    residual=np.asarray(fit.resid)[3:]
    mu,sd=float(residual.mean()),float(residual.std()+1e-6)
    z=(residual-mu)/sd
    xs=np.array([z[i-a.lookback:i] for i in range(a.lookback,len(z))])[:,:,None]
    ys=z[a.lookback:]
    torch.manual_seed(42);torch.set_num_threads(2)
    class ResidualLSTM(nn.Module):
        def __init__(self):
            super().__init__();self.lstm=nn.LSTM(1,8,batch_first=True);self.head=nn.Linear(8,1)
        def forward(self,x):
            h,_=self.lstm(x);return self.head(h[:,-1,:]).squeeze(-1)
    model=ResidualLSTM();opt=torch.optim.Adam(model.parameters(),lr=.01)
    xt=torch.tensor(xs,dtype=torch.float32);yt=torch.tensor(ys,dtype=torch.float32)
    for _ in range(a.epochs):
        opt.zero_grad();loss=nn.functional.mse_loss(model(xt),yt);loss.backward();opt.step()
    model.eval();history=list(z);correction=[]
    with torch.no_grad():
        for _ in test:
            nxt=float(model(torch.tensor(history[-a.lookback:],dtype=torch.float32).reshape(1,a.lookback,1)))
            history.append(nxt);correction.append(nxt*sd+mu)
    logistic=lambda x:1/(1+np.exp(-np.clip(x,-40,40)))
    pred={'last_value':np.repeat(train[-1],len(test)),
          'arima':logistic(base),'arima_lstm':logistic(base+np.asarray(correction))}
    metrics={name:{'mae':float(np.mean(abs(test-v))),'rmse':float(np.sqrt(np.mean((test-v)**2)))}
             for name,v in pred.items()}
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'date':df.date.iloc[-len(test):],'actual':test,**pred}).to_csv(out/'forecast.csv',index=False)
    (out/'forecast_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    torch.save(model.state_dict(),out/'residual_lstm.pt');print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
