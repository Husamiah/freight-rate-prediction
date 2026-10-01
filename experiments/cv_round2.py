import sys, time; sys.path.insert(0,'.')
from experiments.cv_experiments import *
t0=time.time()
print('=== C. Pruning (huber 0.1, no calendar, +extra) ===')
for k,c in {'extra-nocal':{'extra':1,'nocal':1},'extra-nocal+TE':{'extra':1,'nocal':1,'te':1}}.items():
    r=cv(c); print(f"{k:18s} MAE {r['mae']:6.1f} {r['per']} | MAPE {r['mape']:.2f}% | medAPE {r['medape']:.2f}% | it~{r['it']}")
print('=== D. Hyper-parameter grid (extra-nocal) ===')
rows=[]
for nl,mcs in itertools.product([7,15,31,63],[20,60,150]):
    r=cv({'extra':1,'nocal':1,'nl':nl,'mcs':mcs}); rows.append((nl,mcs,r))
    print(f"leaves={nl:<3} min_child={mcs:<4} MAE {r['mae']:6.1f} | MAPE {r['mape']:.2f}% | medAPE {r['medape']:.2f}% | it~{r['it']}")
print(f'elapsed {time.time()-t0:.0f}s')
