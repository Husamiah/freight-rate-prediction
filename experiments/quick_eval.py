import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, lightgbm as lgb
from src.features import FeatureBuilder, load
SEED=42
df=load('data/train_test.csv'); df['rpm']=df.posted_rate/df.distance
tr=df[df.date<'2025-09-01']; ho=df[df.date>='2025-09-01']
fb=FeatureBuilder().fit(tr); Xtr,Xho=fb.transform(tr),fb.transform(ho)
print('features:',Xtr.shape[1],'| train',len(tr),'holdout',len(ho),'| outliers(rpm>5) in holdout',(ho.rpm>5).sum())
def rep(name,pred,y=ho.posted_rate.values):
    ape=np.abs(pred-y)/y; clean=(ho.rpm.values<5)
    print(f'{name:34s} MAE {np.abs(pred-y).mean():7.1f} | RMSE {np.sqrt(((pred-y)**2).mean()):7.1f} | MAPE {ape.mean()*100:5.2f}% | medAPE {np.median(ape)*100:5.2f}% | MAE(clean) {np.abs(pred-y)[clean].mean():6.1f}')
# baseline: median rpm by equipment x distance band
bins=[0,250,500,1000,1500,2000,5000]
tr_b=tr.assign(b=pd.cut(tr.distance,bins)); med=tr_b.groupby(['equipment','b'],observed=True).rpm.median()
ho_b=ho.assign(b=pd.cut(ho.distance,bins)); rep('baseline median rpm(equip,band)', ho_b.apply(lambda r: med[(r.equipment,r.b)],axis=1).values*ho.distance.values)
cats=['pickup_city','delivery_city']
def fit(obj,target,params=None,n=1500):
    p=dict(objective=obj,learning_rate=0.03,num_leaves=31,min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=1.0,random_state=SEED,verbose=-1,n_estimators=n)
    if params: p.update(params)
    m=lgb.LGBMRegressor(**p); m.fit(Xtr,target(tr),eval_set=[(Xho,target(ho))],callbacks=[lgb.early_stopping(100,verbose=False)])
    return m
for obj,extra in [('regression',{}),('huber',{'alpha':0.3}),('regression_l1',{})]:
    m=fit(obj,lambda d:np.log(d.posted_rate),extra); rep(f'LGBM log(rate) {obj} it={m.best_iteration_}',np.exp(m.predict(Xho)))
m=fit('huber',lambda d:np.log(d.rpm),{'alpha':0.3}); rep(f'LGBM log(rpm) huber it={m.best_iteration_}',np.exp(m.predict(Xho))*ho.distance.values)
imp=pd.Series(m.booster_.feature_importance('gain'),Xtr.columns).sort_values(ascending=False); print((imp/imp.sum()*100).round(1).head(12).to_dict())
