import sys,numpy as np,pandas as pd
D=sys.argv[1]
for e in 'ABC':
    r=pd.read_csv(f'{D}/expA.csv'.replace('A',e)) if e!='A' else pd.read_csv(f'{D}/expA.csv')
    dt=np.diff(r.time); dtr=np.diff(r.t_real)
    print(e,len(r),'dt',dt.min(),dt.max(),'dtreal',dtr.min().round(3),dtr.max().round(3))
    for c in ['T1','T2']:
        d=np.diff(r[c]); u=np.unique(np.round(np.abs(d[d!=0]),3))
        # noise: residual from 31-pt moving median
        s=pd.Series(r[c]).rolling(31,center=True).median(); n=(r[c]-s).std()
        print('  ',c,'min',r[c].min(),'max',r[c].max(),'quant steps',u[:4],'noise std',round(n,3),'start',r[c][:5].mean().round(2),'end',r[c].iloc[-5:].mean().round(2))
