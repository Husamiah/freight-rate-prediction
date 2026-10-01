import sys; sys.path.insert(0,'.')
from experiments.cv_experiments import *
def day_feats(X,raw):
    X=X.copy(); g=raw.groupby('date')
    X['mi_day']=g.market_index.transform('mean').values; X['qs_day']=g.quote_signal.transform('mean').values
    X['mi_dev']=X.market_index-X.mi_day; X['qs_dev_day']=X.quote_signal-X.qs_day; return X
def go(name,addday,adddow):
    out=[]
    for _,s,e in FOLDS:
        tr=df[df.date<s]; va=df[(df.date>=s)&(df.date<e)]; fb=FeatureBuilder().fit(tr)
        Xtr,Xva=extra(fb.transform(tr)),extra(fb.transform(va))
        keep=[c for c in Xtr.columns if c not in CAL]+(['dow'] if adddow else []); Xtr,Xva=Xtr[keep],Xva[keep]
        if addday: Xtr,Xva=day_feats(Xtr,tr),day_feats(Xva,va)
        p=dict(objective='huber',alpha=0.1,learning_rate=0.05,num_leaves=7,min_child_samples=150,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,random_state=SEED,verbose=-1,n_estimators=1500)
        pr=np.exp(lgb.LGBMRegressor(**p).fit(Xtr,tr.lrpm).predict(Xva))*va.distance.values; y=va.posted_rate.values; ape=np.abs(pr-y)/y
        out.append((np.abs(pr-y).mean(),ape.mean()*100,np.median(ape)*100))
    a=np.mean(out,0); print(f'{name:28s} MAE {a[0]:6.1f} [{out[0][0]:.1f}, {out[1][0]:.1f}] | MAPE {a[1]:.2f}% | medAPE {a[2]:.2f}%')
go('current (no day feats)',False,False); go('+ day-level market feats',True,False); go('+ day feats + dow',True,True); go('+ dow only',False,True)
