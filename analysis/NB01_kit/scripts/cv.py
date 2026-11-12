# Validación cruzada A<->C (lineales) y A+B<->C (físico entero vs fraccionario)
import sys, json, time, numpy as np, pandas as pd
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
A, B, C = load(f'{D}/expA.csv',2710), load(f'{D}/expB.csv',2710), load(f'{D}/expC.csv',3600)
dev = lambda df,c: df[c].values - df[c].values[:5].mean()
rows = []; T0 = time.time()
# ---- lineales: G11 (T1) y G21 (T2) con Q1, A <-> C
for ch, tc in [('G11','T1'), ('G21','T2')]:
    for idn, dfi, vdn, dfv in [('A', A, 'C', C), ('C', C, 'A', A)]:
        u, y = dfi.Q1.values, dev(dfi, tc); uv, yv = dfv.Q1.values, dev(dfv, tc)
        full = slice(0, len(y))
        # arranque 2 puntos con el primer escalón de A
        g2 = two_point(A.time.values[:910], A.Q1.values[:910], dev(A, tc)[:910], 10)
        for name in MODELS:
            p, se = fit(name, u, y, full, g2)
            yh = MODELS[name]['fn'](p, u); yvh = MODELS[name]['fn'](p, uv)
            n = len(y); sse = float(((yh-y)**2).sum())
            par = dict(zip(MODELS[name]['names'], p)); par.update({f'se_{k}': s for k, s in zip(MODELS[name]['names'], se)})
            rows.append(dict(canal=ch, id=idn, val=vdn, modelo=name, npar=len(p), RMSE_id=np.sqrt(sse/n),
                             AIC=n*np.log(sse/n)+2*len(p), RMSE_val=np.sqrt(np.mean((yvh-yv)**2)), **par))
            print(ch, idn, name, round(rows[-1]['RMSE_id'],3), round(rows[-1]['RMSE_val'],3), {k: round(v,3) for k,v in par.items() if not k.startswith('se')}, f'{time.time()-T0:.0f}s', flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/cv_lineales.csv', index=False)
# ---- físico 2x2: entero vs sensor fraccionario
PH = ['U','Us','alpha1','alpha2','tau_s']
def pred(th, df, frac):
    p = dict(PHYS); p.update(zip(PH, th[:5])); p['sensor_order'] = th[5] if frac else 1.0
    return phys_model(df.Q1.values, df.Q2.values, p, nsub=2, Ta=df[['T1','T2']].values[:5].mean())
def resid(th, dfs, frac):
    out = []
    for df in dfs:
        T1, T2 = pred(th, df, frac); out += [T1-df.T1.values, T2-df.T2.values]
    return np.concatenate(out)
prow = []
for idn, dfi, vdn, dfv in [('A+B', [A, B], 'C', [C]), ('C', [C], 'A+B', [A, B])]:
    for frac in [False, True]:
        lb, ub, x0 = [1,1,1e-3,1e-3,1], [30,100,0.05,0.05,150], [8,15,0.01,0.0075,15]
        if frac: lb, ub, x0 = lb+[0.3], ub+[1.5], x0+[0.9]
        s = least_squares(resid, x0, bounds=(lb,ub), x_scale='jac', max_nfev=200, args=(dfi, frac))
        se = np.sqrt(np.diag(2*s.cost/(len(s.fun)-len(s.x))*np.linalg.pinv(s.jac.T@s.jac)))
        names = PH + (['beta'] if frac else [])
        rv = resid(s.x, dfv, frac); n = len(s.fun)
        d = dict(id=idn, val=vdn, modelo='Físico fracc.' if frac else 'Físico entero', npar=len(s.x),
                 RMSE_id=np.sqrt(np.mean(s.fun**2)), AIC=n*np.log(np.mean(s.fun**2))+2*len(s.x), RMSE_val=np.sqrt(np.mean(rv**2)),
                 **dict(zip(names, s.x)), **{f'se_{k}': v for k, v in zip(names, se)})
        # RMSE val por salida
        k = 0
        for df in dfv:
            m = len(df); d.setdefault('RMSE_val_T1', []); d.setdefault('RMSE_val_T2', [])
            d['RMSE_val_T1'].append(rv[k:k+m]); d['RMSE_val_T2'].append(rv[k+m:k+2*m]); k += 2*m
        d['RMSE_val_T1'] = float(np.sqrt(np.mean(np.concatenate(d['RMSE_val_T1'])**2)))
        d['RMSE_val_T2'] = float(np.sqrt(np.mean(np.concatenate(d['RMSE_val_T2'])**2)))
        prow.append(d); print({k: (round(v,4) if isinstance(v,float) else v) for k,v in d.items()}, f'{time.time()-T0:.0f}s', flush=True)
        np.save(f'{OUT}/phys_{idn.replace("+","")}_{int(frac)}.npy', s.x)
pd.DataFrame(prow).to_csv(f'{OUT}/cv_fisico.csv', index=False)
print('fin', time.time()-T0)
