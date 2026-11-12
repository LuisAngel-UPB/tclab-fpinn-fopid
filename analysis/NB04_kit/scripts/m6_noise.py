"""M6 (revisión CEP): Ms, PM y sensibilidad al ruido max|C S| de los controladores, sobre OE-LS entero y fPINN."""
import json, sys, numpy as np, pandas as pd
from scipy.optimize import brentq
ctrl = json.load(open(sys.argv[1]))["controllers"]; sur = json.load(open(sys.argv[2])); out = sys.argv[3]
ph = sur["phys"]; m, cp, A, As, eps, sig = ph["m"], ph["cp"], ph["A"], ph["As"], ph["eps"], ph["sigma"]
Tamb = sur.get("T_AMB_NB01", 28.0) + 273.15; Tf, h = 10.0, 1.0
w = np.logspace(-5, np.log10(np.pi / h), 6000)
def F(T, Q, p):
    T1, T2 = T
    f1 = p["U"]*A*(Tamb-T1) + eps*sig*A*(Tamb**4-T1**4) + p["Us"]*As*(T2-T1) + eps*sig*As*(T2**4-T1**4) + p["alpha1"]*Q[0]
    f2 = p["U"]*A*(Tamb-T2) + eps*sig*A*(Tamb**4-T2**4) + p["Us"]*As*(T1-T2) + eps*sig*As*(T1**4-T2**4) + p["alpha2"]*Q[1]
    return np.array([f1, f2])
def plant(p, Tbar):
    T1 = Tbar + 273.15
    T2 = brentq(lambda T2: F([T1, T2], [0, 0], p)[1], Tamb - 1, T1 + 1)  # Q2 = 0
    Gam = np.zeros((2, 2)); d = 1e-4
    for j in range(2):
        e = np.zeros(2); e[j] = d
        Gam[:, j] = -(F(np.array([T1, T2]) + e, [0, 0], p) - F(np.array([T1, T2]) - e, [0, 0], p)) / (2*d)
    a, b = p["alpha"], p["beta"]; s = 1j*w
    G = np.empty(len(w), complex)
    for k, sk in enumerate(s):
        M = m*cp*sk**a*np.eye(2) + Gam
        G[k] = p["alpha1"]*np.linalg.inv(M)[0, 0] / ((p["tau_s"]*sk)**b + 1) if b == 1 else p["alpha1"]*np.linalg.inv(M)[0, 0] / (p["tau_s"]**b*sk**b + 1)
    return G*np.exp(-s*h/2)
def C(q):
    s = 1j*w
    return q["Kp"] + q["Ki"]/s**q["lam"] + q["Kd"]*s**q["mu"]/(Tf*s + 1)
rows = []
for mod in ["PhysInt_OE", "fPINN"]:
    p = sur["models"][mod]
    for Tbar in [35, 50, 65]:
        G = plant(p, Tbar)
        for cid in ["P", "B1", "B2", "B3", "B4"]:
            c = C(ctrl[cid]["p"]); L = c*G; S = 1/(1+L)
            i = np.where(np.diff(np.sign(np.abs(L)-1)))[0]
            pm = 180 + np.degrees(np.angle(L[i[0]])) if len(i) else np.nan
            rows.append(dict(model=mod, op=Tbar, id=cid, Ms=np.abs(S).max(), PM=pm,
                             max_CS=np.abs(c*S).max(), C_nyq=np.abs(c[-1]), CS_nyq=np.abs(c[-1]*S[-1])))
df = pd.DataFrame(rows); df.to_csv(f"{out}/ruido_por_op.csv", index=False)
wc = df.groupby(["model", "id"]).agg(Ms=("Ms", "max"), PM=("PM", "min"), max_CS=("max_CS", "max"), C_nyq=("C_nyq", "max")).reset_index()
wc.to_csv(f"{out}/ruido_peor_caso.csv", index=False)
pd.set_option("display.width", 200); print(wc.round(3).to_string())
# Ganancia del derivativo puro a alta frecuencia (mu>1 => crece como w^(mu-1))
for cid in ["P", "B3", "B4"]:
    q = ctrl[cid]["p"]
    print(cid, "|Kd s^mu/(Tf s+1)| en w=0.1, 1, pi, 10 rad/s:", [round(abs(q["Kd"]*(1j*x)**q["mu"]/(Tf*1j*x+1)), 1) for x in [0.1, 1, np.pi, 10]])
