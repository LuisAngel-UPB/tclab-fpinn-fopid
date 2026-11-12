# Físico 2x2 multiarranque; C->A con alpha2 fijo; análisis de residuos de modo común (ambiente)
import sys, time, numpy as np, pandas as pd
from scipy.optimize import least_squares
sys.path.insert(0, sys.argv[2]); from lib import *
D, OUT = sys.argv[1], sys.argv[3]
def load(f, tend):
    r = pd.read_csv(f); tr = r.time.values; t = np.arange(0, tend, 1.0)
    df = pd.DataFrame({'time': t})
    for c in ['T1','T2']: df[c] = np.interp(t, tr, r[c])
    idx = np.clip(np.searchsorted(tr, t, side='right')-1, 0, len(r)-1)
    for c in ['Q1','Q2']: df[c] = r[c].values[idx].astype(float)
    return df
E = dict(A=load(f'{D}/expA.csv',2710), B=load(f'{D}/expB.csv',2710), C=load(f'{D}/expC.csv',3600))
PH = ['U','Us','alpha1','alpha2','tau_s']
def pred(th, df, frac, a2=None):
    th = list(th)
    if a2 is not None: th = th[:3] + [a2] + th[3:]
    p = dict(PHYS); p.update(zip(PH, th[:5])); p['sensor_order'] = th[5] if frac else 1.0
    return phys_model(df.Q1.values, df.Q2.values, p, nsub=2, Ta=df[['T1','T2']].values[:5].mean())
def resid(th, dfs, frac, a2=None):
    return np.concatenate([np.r_[T1-df.T1.values, T2-df.T2.values] for df in dfs for T1, T2 in [pred(th, df, frac, a2)]])
rng = np.random.default_rng(1); rows = []; T0 = time.time(); best = {}
for idn, ids, vds, fixa2 in [('A+B', 'AB', 'C', False), ('C', 'C', 'A', True)]:
    a2 = best.get(('A+B', False), [None]*4)[3] if fixa2 else None
    for frac in [False, True]:
        lb, ub, x0 = [1,1,1e-3,1e-3,1], [30,100,0.05,0.05,150], [8,15,0.01,0.0075,15]
        if fixa2: lb, ub, x0 = [v for i, v in enumerate(lb) if i != 3], [v for i, v in enumerate(ub) if i != 3], [v for i, v in enumerate(x0) if i != 3]
        if frac: lb, ub, x0 = lb+[0.3], ub+[1.5], x0+[0.9]
        sols = []
        for k in range(4):
            x = np.array(x0) * (1 if k == 0 else rng.uniform(0.6, 1.5, len(x0)))
            if frac and k > 0: x[-1] = [0.7, 1.0, 1.2][k-1]
            x = np.clip(x, np.array(lb)+1e-6, np.array(ub)-1e-6)
            sols.append(least_squares(resid, x, bounds=(lb, ub), x_scale='jac', max_nfev=200, args=([E[c] for c in ids], frac, a2)))
        s = min(sols, key=lambda r: r.cost)
        se = np.sqrt(np.diag(2*s.cost/(len(s.fun)-len(s.x))*np.linalg.pinv(s.jac.T@s.jac)))
        names = [n for n in PH if not (fixa2 and n == 'alpha2')] + (['beta'] if frac else [])
        best[(idn, frac)] = s.x
        d = dict(id=idn, val=vds, modelo='Físico fracc.' if frac else 'Físico entero', npar=len(s.x),
                 RMSE_id=np.sqrt(np.mean(s.fun**2)), AIC=len(s.fun)*np.log(np.mean(s.fun**2))+2*len(s.x))
        for v in vds:
            T1, T2 = pred(s.x, E[v], frac, a2)
            d[f'RMSE_val_{v}_T1'] = np.sqrt(np.mean((T1-E[v].T1.values)**2)); d[f'RMSE_val_{v}_T2'] = np.sqrt(np.mean((T2-E[v].T2.values)**2))
        d.update(dict(zip(names, s.x))); d.update({f'se_{k}': v for k, v in zip(names, se)})
        if fixa2: d['alpha2'] = a2
        rows.append(d); print({k: (round(float(v),4) if not isinstance(v,str) else v) for k,v in d.items()}, f'{time.time()-T0:.0f}s', flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/cv_fisico.csv', index=False)
# residuos del físico entero (A+B) en los tres experimentos -> modo común = ambiente
th = best[('A+B', False)]; R = {}
for e, df in E.items():
    T1, T2 = pred(th, df, False); R[e] = pd.DataFrame({'time': df.time, 'r1': df.T1-T1, 'r2': df.T2-T2, 'Q1': df.Q1, 'Q2': df.Q2})
    R[e].to_csv(f'{OUT}/residuos_fisico_{e}.csv', index=False)
    print(e, 'corr(r1,r2)=', round(np.corrcoef(R[e].r1, R[e].r2)[0,1], 2), 'std r1,r2', round(R[e].r1.std(),2), round(R[e].r2.std(),2))
np.save(f'{OUT}/best.npy', np.array([np.r_[best[k], [np.nan]*(6-len(best[k]))] for k in best]))
