import sys, time; sys.path.insert(0,'.')
from experiments.cv_experiments import *
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
t0=time.time()
def prep(start,end):
    tr=df[df.date<start]; va=df[(df.date>=start)&(df.date<end)]; fb=FeatureBuilder().fit(tr)
    Xtr,Xva=extra(fb.transform(tr)).drop(columns=CAL),extra(fb.transform(va)).drop(columns=CAL); return tr,va,Xtr,Xva
def lgbm(Xtr,ytr,Xva,**kw):
    p=dict(objective='huber',alpha=0.1,learning_rate=0.05,num_leaves=7,min_child_samples=150,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=1.0,random_state=SEED,verbose=-1,n_estimators=1500); p.update(kw)
    return lgb.LGBMRegressor(**p).fit(Xtr,ytr).predict(Xva)
def spline_ridge(Xtr,ytr,Xva):
    num=['log_distance','distance','haversine','quote_signal','market_index','weight','distance_ratio','qs_dev']
    Xt,Xv=Xtr.copy(),Xva.copy()
    for X in (Xt,Xv): X['market_index']=X['market_index'].fillna(Xtr['market_index'].median())
    for e in range(3):
        for X in (Xt,Xv): X[f'eq{e}']=(X.equipment==e).astype(float)
    # equipment interactions with log_distance
    for e in range(3):
        for X in (Xt,Xv): X[f'eq{e}_ld']=X[f'eq{e}']*X.log_distance
    sp=ColumnTransformer([('s',SplineTransformer(n_knots=8,degree=3),num)],remainder='passthrough')
    mdl=make_pipeline(sp,StandardScaler(),RidgeCV(alphas=[0.1,1,10,100])); mdl.fit(Xt.drop(columns=['pickup_city','delivery_city']),ytr)
    return mdl.predict(Xv.drop(columns=['pickup_city','delivery_city']))
def hgb(Xtr,ytr,Xva): return HistGradientBoostingRegressor(loss='absolute_error',learning_rate=0.05,max_leaf_nodes=8,min_samples_leaf=150,max_iter=800,random_state=SEED).fit(Xtr,ytr).predict(Xva)
def score(pred_l,va):
    p=np.exp(pred_l)*va.distance.values; y=va.posted_rate.values; ape=np.abs(p-y)/y; return np.abs(p-y).mean(),ape.mean()*100,np.median(ape)*100
models={'LGBM huber (leaves7)':lambda a,b,c:lgbm(a,b,c),'LGBM linear_tree':lambda a,b,c:lgbm(a,b,c,linear_tree=True,n_estimators=600,learning_rate=0.03),'HistGB (L1)':hgb,'Spline+Ridge':spline_ridge}
res={k:[] for k in list(models)+['Blend LGBM+Spline','Blend LGBM+LGBM_lin+Spline']}
for _,s,e in FOLDS:
    tr,va,Xtr,Xva=prep(s,e); P={k:f(Xtr,tr.lrpm,Xva) for k,f in models.items()}
    P['Blend LGBM+Spline']=(P['LGBM huber (leaves7)']+P['Spline+Ridge'])/2
    P['Blend LGBM+LGBM_lin+Spline']=(P['LGBM huber (leaves7)']+P['LGBM linear_tree']+P['Spline+Ridge'])/3
    for k,v in P.items(): res[k].append(score(v,va))
print('=== E. Model comparison (mean of 2 CV folds) ===')
for k,v in res.items():
    a=np.mean(v,axis=0); print(f"{k:30s} MAE {a[0]:6.1f} [{v[0][0]:.1f}, {v[1][0]:.1f}] | MAPE {a[1]:.2f}% | medAPE {a[2]:.2f}%")
print(f'elapsed {time.time()-t0:.0f}s')
