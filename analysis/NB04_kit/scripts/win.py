# Métricas por ventana transitoria: kit vs sustituto (mismo escenario NB03, misma ley)
import sys, json, os, numpy as np, pandas as pd
from scipy.optimize import fsolve
from scipy.signal import butter, filtfilt
from scipy import stats
KIT, NB03, NB02, OUT = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
gl=lambda a,n: np.cumprod(np.r_[1.0,1.0-(a+1.0)/np.arange(1,n)])
C=json.load(open(f'{NB03}/controladores_NB04.json'))['controllers']
IDS=['P','B1','B2','B3','B4']; OPS=[35.,50.,65.]; W=300  # ventana [s]
class M:
    def __init__(s,th,ph,Ta):
        s.th=dict(th); s.ph=ph; s.Ta=Ta; s.mcp=ph['m']*ph['cp']
    def F(s,x1,x2,q1,q2):
        th,p=s.th,s.ph; TaK=s.Ta+273.15; es,A,As=p['eps']*p['sigma'],p['A'],p['As']; T1,T2=TaK+x1,TaK+x2
        ex=th['Us']*As*(T2-T1)+es*As*(T2**4-T1**4)
        return (th['U']*A*(TaK-T1)+es*A*(TaK**4-T1**4)+ex+th['alpha1']*q1, th['U']*A*(TaK-T2)+es*A*(TaK**4-T2**4)-ex+th['alpha2']*q2)
    def eq(s,T):
        x1=T-s.Ta; v=fsolve(lambda v: s.F(x1,v[1],v[0],0.0),[x1/.75,.3*x1]); return v[0],np.array([x1,v[1],x1,v[1]])
def sim(m,p,Tf,T,N=1800,Nr=900):
    ueq,xb=m.eq(T); al,be=m.th['alpha'],m.th['beta']; ca=1/(m.mcp); cb=(1/m.th['tau_s'])**be
    wa,wb=gl(al,N+2),gl(be,N+2); wI,wD=gl(-p['lam'],N+1),gl(p['mu'],N+1); a=Tf/(Tf+1)
    H=np.zeros((N+1,2)); S=np.zeros((N+1,2)); E=np.zeros(N); Y=np.zeros(N); up=vp=ueq; d=0
    nz=np.random.default_rng(123).normal(0,.03,N); e_=np.zeros(N); u_=np.zeros(N)
    for k in range(N):
        y=T+S[k,0]; ym=y+nz[k]; r=T+(5 if k>=10 else 0); e=r-ym
        et=e+p['Kt']*(up-vp); E[k]=et; Y[k]=ym-T
        I=np.dot(wI[:k+1],E[k::-1]); Dm=np.dot(wD[:k+1],Y[k::-1]); d=a*d+(1-a)*Dm
        v=ueq+p['Kp']*e+p['Ki']*I-p['Kd']*d; u=min(max(v,0),100); up,vp=u,v; e_[k]=r-y; u_[k]=u
        x=xb+np.r_[H[k],S[k]]; f1,f2=m.F(x[0],x[1],u,40.0 if k>=Nr else 0.0)
        S[k+1]=cb*(H[k]-S[k])-wb[1:k+2]@S[k::-1]; H[k+1]=ca*np.array([f1,f2])-wa[1:k+2]@H[k::-1]
    return e_,u_
js=json.load(open(f'{NB02}/surrogates/ABC_const_s0.json'))
mods={'fPINN':M(js['models']['fPINN'],js['phys'],js['T_AMB_NB01']),'IntegerPINN':M(js['models']['IntegerPINN'],js['phys'],js['T_AMB_NB01'])}
for mm in mods.values(): mm.th.setdefault('Us',mm.th['U'])
def win(e,t,Nr=900):
    m=lambda a,b: np.abs(e[(t>=a)&(t<b)]).sum()
    return dict(IAEr_w=m(10,10+W), IAEd_w=m(Nr,Nr+W), emax_d=np.abs(e[t>=Nr][:W]).max())
rows=[]
for cid in IDS:
    for mn in (['fPINN','IntegerPINN'] if cid=='B3' else ['fPINN']):
        for T in OPS:
            e,u=sim(mods[mn],C[cid]['p'],C[cid]['Tf'],T); rows.append(dict(src='sust_'+mn,id=cid,op=T,rep=0,**win(e,np.arange(1800))))
b=butter(2,0.05/0.5); plan=pd.read_csv(f'{KIT}/plan_NB04.csv')
for p in plan.itertuples():
    D=pd.read_csv(f'{KIT}/corridas/{p.archivo}'); D['T1f']=filtfilt(*b,D.T1.values)
    for T in OPS:
        E=D[(D.fase=='ensayo')&(D.op==T)]; e=(E.r-E.T1f).values; t=E.tau.values
        # piso de regulación: media |e| en los 300 s previos a la perturbación (600–900 s), mismo largo de ventana
        fl=np.abs(e[(t>=600)&(t<900)]).sum()
        rows.append(dict(src='kit',id=p.id,op=T,rep=p.rep,piso=fl,**win(e,t)))
R=pd.DataFrame(rows); R.to_csv(f'{OUT}/metricas_ventana_300s.csv',index=False)
K=R[R.src=='kit'].groupby(['id','rep'])[['IAEr_w','IAEd_w','piso']].sum().reset_index()
K['IAEd_w_neto']=K.IAEd_w-K.piso
S_=R[R.src!='kit'].groupby(['src','id'])[['IAEr_w','IAEd_w']].sum().reset_index()
out=[]
for cid in IDS:
    k=K[K.id==cid]; s=S_[(S_.id==cid)&(S_.src==('sust_IntegerPINN' if cid=='B3' else 'sust_fPINN'))].iloc[0]
    out.append(dict(id=cid,IAEr_w_kit=k.IAEr_w.mean(),IAEr_w_kit_sd=k.IAEr_w.std(),IAEr_w_sust=s.IAEr_w,
                    IAEd_w_kit=k.IAEd_w.mean(),IAEd_w_kit_sd=k.IAEd_w.std(),IAEd_w_sust=s.IAEd_w,piso_kit=k.piso.mean(),IAEd_w_neto_kit=k.IAEd_w_neto.mean()))
O=pd.DataFrame(out); O['dIAEr_pct']=100*(O.IAEr_w_kit/O.IAEr_w_sust-1); O['dIAEd_pct']=100*(O.IAEd_w_kit/O.IAEd_w_sust-1)
O.to_csv(f'{OUT}/resumen_ventana_300s.csv',index=False)
pd.set_option('display.width',250); print(O.round(1).to_string(index=False))
print(S_.round(1).to_string(index=False))
pt=[]
a=K[K.id=='P'].set_index('rep')
for cid in ['B1','B2','B3','B4']:
    c=K[K.id==cid].set_index('rep')
    for col in ['IAEr_w','IAEd_w','IAEd_w_neto']:
        pt.append(dict(comp=cid,met=col,P=a[col].mean(),comp_m=c[col].mean(),mejora_pct=100*(c[col].mean()-a[col].mean())/c[col].mean(),p=stats.ttest_rel(a[col],c[col]).pvalue))
PT=pd.DataFrame(pt); PT.to_csv(f'{OUT}/pruebas_pareadas_ventana_300s.csv',index=False); print(PT.round(3).to_string(index=False))
