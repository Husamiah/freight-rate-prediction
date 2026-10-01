import sys, warnings, itertools; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
from src.features import FeatureBuilder, load
SEED=42
df=load('data/train_test.csv'); df['rpm']=df.posted_rate/df.distance; df['lrpm']=np.log(df.rpm); df['lane']=df.pickup+'>'+df.delivery
FOLDS=[('F1 May-Jun','2025-05-01','2025-07-01'),('F2 Jul-Aug','2025-07-01','2025-09-01')]
HOLD=('Holdout Sep-Oct','2025-09-01','2025-11-01')

def te_stats(tr,key,m=20):
    g=tr.groupby(key).lrpm.agg(['sum','count']); mu=tr.lrpm.mean()
    return ((g['sum']+m*mu)/(g['count']+m)-mu), mu
def add_te(Xtr,tr,Xap,ap,oof):
    """lane/pickup/delivery target encoding of log-rpm. OOF on train to avoid leakage."""
    cols={'pickup':'te_pickup','delivery':'te_delivery','lane':'te_lane'}
    Xtr,Xap=Xtr.copy(),Xap.copy()
    for k,name in cols.items():
        s,_=te_stats(tr,k); Xap[name]=ap[k].map(s).values
        o=np.full(len(tr),np.nan)
        for a,b in KFold(5,shuffle=True,random_state=SEED).split(tr):
            sa,_=te_stats(tr.iloc[a],k); o[b]=tr.iloc[b][k].map(sa).values
        Xtr[name]=o
    # unseen lane -> fall back to mean of pickup & delivery effects
    for X in (Xtr,Xap): X['te_lane']=X['te_lane'].fillna((X['te_pickup']+X['te_delivery'])/2)
    return Xtr,Xap
def extra(X):
    X=X.copy(); X['qs_dev']=(X.quote_signal-2.06).abs(); X['qs_x_logd']=X.quote_signal*X.log_distance
    X['w_x_equip']=X.weight*(X.equipment+1); return X
CAL=['dow','is_weekend','dom','days_to_month_end','days_since_holiday','days_to_holiday','near_holiday']

def run(cfg,start,end,fixed_n=None):
    tr=df[df.date<start]; va=df[(df.date>=start)&(df.date<end)]
    fb=FeatureBuilder().fit(tr); Xtr,Xva=fb.transform(tr),fb.transform(va)
    if cfg.get('te'): Xtr,Xva=add_te(Xtr,tr,Xva,va,True)
    if cfg.get('extra'): Xtr,Xva=extra(Xtr),extra(Xva)
    if cfg.get('nocal'): Xtr,Xva=Xtr.drop(columns=CAL),Xva.drop(columns=CAL)
    p=dict(objective=cfg.get('obj','huber'),alpha=cfg.get('alpha',0.1),learning_rate=0.03,num_leaves=cfg.get('nl',31),
           min_child_samples=cfg.get('mcs',40),subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=1.0,
           random_state=SEED,verbose=-1,n_estimators=fixed_n or 2000)
    m=lgb.LGBMRegressor(**p)
    if fixed_n: m.fit(Xtr,tr.lrpm)
    else: m.fit(Xtr,tr.lrpm,eval_set=[(Xva,va.lrpm)],callbacks=[lgb.early_stopping(100,verbose=False)])
    pred=np.exp(m.predict(Xva))*va.distance.values; y=va.posted_rate.values; ape=np.abs(pred-y)/y
    return dict(mae=np.abs(pred-y).mean(),mape=ape.mean()*100,medape=np.median(ape)*100,it=m.best_iteration_ if not fixed_n else fixed_n),m,Xtr

def cv(cfg):
    rs=[run(cfg,s,e)[0] for _,s,e in FOLDS]
    return dict(mae=np.mean([r['mae'] for r in rs]),mape=np.mean([r['mape'] for r in rs]),medape=np.mean([r['medape'] for r in rs]),it=int(np.mean([r['it'] for r in rs])),per=[round(r['mae'],1) for r in rs])

if __name__=='__main__':
    print('=== A. Feature experiments (alpha=0.1, mean over 2 CV folds) ===')
    exps={'base':{}, '+extra':{'extra':1}, '+TE lane/city':{'te':1}, '+TE+extra':{'te':1,'extra':1}, 'base-calendar':{'nocal':1}, '+TE+extra-calendar':{'te':1,'extra':1,'nocal':1}}
    for k,c in exps.items():
        r=cv(c); print(f"{k:22s} MAE {r['mae']:6.1f} {r['per']} | MAPE {r['mape']:.2f}% | medAPE {r['medape']:.2f}% | it~{r['it']}")
    print('=== B. Huber alpha / L1 (base features) ===')
    for a in [0.02,0.05,0.1,0.3]:
        r=cv({'alpha':a}); print(f"huber alpha={a:<5} MAE {r['mae']:6.1f} | MAPE {r['mape']:.2f}% | medAPE {r['medape']:.2f}% | it~{r['it']}")
    r=cv({'obj':'regression_l1'}); print(f"l1                    MAE {r['mae']:6.1f} | MAPE {r['mape']:.2f}% | medAPE {r['medape']:.2f}% | it~{r['it']}")
