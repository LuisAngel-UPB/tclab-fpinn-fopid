"""M5 (revisión CEP): ¿la cuantización del ADC (0,32 °C) reproduce el piso de regulación del kit?
Planta virtual = modelo físico entero OE-LS (A+B+C), no lineal 2x2; ley de control = controladores_NB04.json.
Escenarios de medición: 'blanco' (ruido 0,03 °C, como el sustituto) y 'cuant' (ruido 0,03 °C + cuantización 0,32 °C)."""
import json, sys, numpy as np, pandas as pd
from scipy.signal import butter, filtfilt
from scipy.optimize import brentq
ctrl = json.load(open(sys.argv[1]))["controllers"]; sur = json.load(open(sys.argv[2])); out = sys.argv[3]
p = sur["models"]["PhysInt_OE"]; ph = sur["phys"]
m, cp, A, As, eps, sig = ph["m"], ph["cp"], ph["A"], ph["As"], ph["eps"], ph["sigma"]
TAMB = 29.0; h = 1.0; Tf = 10.0; N = 1800; SUB = 10
def F(T1, T2, Q1, Q2):
    a1, a2 = T1 + 273.15, T2 + 273.15; ta = TAMB + 273.15
    f1 = p["U"]*A*(ta-a1) + eps*sig*A*(ta**4-a1**4) + p["Us"]*As*(a2-a1) + eps*sig*As*(a2**4-a1**4) + p["alpha1"]*Q1
    f2 = p["U"]*A*(ta-a2) + eps*sig*A*(ta**4-a2**4) + p["Us"]*As*(a1-a2) + eps*sig*As*(a1**4-a2**4) + p["alpha2"]*Q2
    return f1/(m*cp), f2/(m*cp)
def equilibrium(Tbar):
    T2 = brentq(lambda T2: F(Tbar, T2, 0, 0)[1], TAMB - 1, Tbar + 1)
    Q1 = brentq(lambda Q: F(Tbar, T2, Q, 0)[0], -50, 200)
    return T2, Q1
def gl_w(a, n):
    w = np.ones(n)
    for i in range(1, n): w[i] = w[i-1]*(1 - (1 + a)/i)
    return w
B, Af = butter(2, 0.05/(0.5/h))
def run(cid, Tbar, scen, seed):
    q = ctrl[cid]["p"]; rng = np.random.default_rng(seed)
    wI, wD = gl_w(-q["lam"], N), gl_w(q["mu"], N)
    T2, u0 = equilibrium(Tbar); TH1, TH2, TS1, TS2 = Tbar, T2, Tbar, T2
    y0 = Tbar; a = Tf/(Tf + h)
    et = np.zeros(N); yd = np.zeros(N); ys = np.zeros(N); us = np.zeros(N); r = np.zeros(N)
    u_prev = v_prev = u0; d = 0.0
    for k in range(N):
        y = TS1 + rng.normal(0, 0.03)
        if scen == "cuant": y = np.round(y/0.32)*0.32
        rk = Tbar + (5.0 if k*h >= 10 else 0.0); Q2 = 40.0 if k*h >= 900 else 0.0
        e = rk - y; et[k] = e + q["Kt"]*(u_prev - v_prev); yd[k] = y - y0
        I = h**q["lam"]*np.dot(wI[:k+1], et[k::-1]); D = h**(-q["mu"])*np.dot(wD[:k+1], yd[k::-1])
        d = a*d + (1 - a)*D
        v = u0 + q["Kp"]*e + q["Ki"]*I - q["Kd"]*d; u = min(max(v, 0.0), 100.0)
        ys[k] = TS1; us[k] = u; r[k] = rk; u_prev, v_prev = u, v
        for _ in range(SUB):
            d1, d2 = F(TH1, TH2, u, Q2); dt = h/SUB
            TH1 += dt*d1; TH2 += dt*d2; TS1 += dt*(TH1 - TS1)/p["tau_s"]; TS2 += dt*(TH2 - TS2)/p["tau_s"]
    # métricas como NB04: sobre la medición filtrada fuera de línea; aquí la medición "vista" se reconstruye
    return ys, us, r
rows = []
for scen in ["blanco", "cuant"]:
    for cid in ["P", "B1", "B2", "B3", "B4"]:
        for Tbar in [35, 50, 65]:
            for seed in range(3):
                rng = np.random.default_rng(100 + seed)
                ys, us, r = run(cid, Tbar, scen, 100 + seed)
                # medición tal como la registraría el kit (misma regla del escenario), luego filtrada como en NB04
                ym = ys + rng.normal(0, 0.03, N)
                if scen == "cuant": ym = np.round(ym/0.32)*0.32
                yf = filtfilt(B, Af, ym); e = np.abs(r - yf)
                ss = slice(600, 900)  # régimen antes de la perturbación
                rows.append(dict(scen=scen, id=cid, op=Tbar, seed=seed, IAE_r=e[:900].sum()*h, IAE_d=e[900:].sum()*h,
                                 TV=np.abs(np.diff(us)).sum(), std_ss=np.std(yf[ss] - r[ss]),
                                 IAE_floor_300=e[ss].sum()*h))
df = pd.DataFrame(rows); df.to_csv(f"{out}/sim_cuantizacion_por_tramo.csv", index=False)
g = df.groupby(["scen", "id"]).agg(IAE_r=("IAE_r", "mean"), IAE_d=("IAE_d", "mean"), TV=("TV", "mean")).reset_index()
g[["IAE_r", "IAE_d", "TV"]] *= 3  # suma sobre 3 OP
pd.set_option("display.width", 200)
print(g.round(1).to_string())
print(df.groupby(["scen", "op"])[["std_ss", "IAE_floor_300"]].mean().round(3))
print(df.groupby(["scen", "id"])[["std_ss"]].mean().round(3))
