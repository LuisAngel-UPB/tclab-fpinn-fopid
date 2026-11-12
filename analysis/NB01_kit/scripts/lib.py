import numpy as np, pandas as pd
from scipy.optimize import least_squares
TS=1.0
N_STARTS=5
PHYS = dict(Ta=23.0, U=4.0, Us=10.0, A=12e-4, As=2e-4, m=4e-3, cp=500.0, eps=0.9,
            alpha1=0.01, alpha2=0.0075, tau_s=20.0, sensor_order=1.0)

def gl_weights(a, n):
    """Pesos de Grünwald-Letnikov g_j para el orden a (g_0 = 1)."""
    return np.cumprod(np.r_[1.0, 1.0 - (a + 1.0) / np.arange(1, n)])

def phys_model(Q1, Q2, p, ts=TS, nsub=10, Ta=None):
    """Modelo físico del TCLab. Convención: y_k se lee en t_k, antes de aplicar Q_k."""
    sig = 5.67e-8
    TaC = p['Ta'] if Ta is None else Ta
    TaK = TaC + 273.15
    U, Us, A, As, mcp, eps = p['U'], p['Us'], p['A'], p['As'], p['m'] * p['cp'], p['eps']
    a1, a2, tau_s, beta = p['alpha1'], p['alpha2'], p['tau_s'], p.get('sensor_order', 1.0)
    N = len(Q1); h = ts / nsub
    TH1 = TH2 = TaK
    w = np.zeros((N, 2)); e = np.zeros((N, 2))
    g = gl_weights(beta, N); c = tau_s * ts ** (-beta); den = c * g[0] + 1.0
    for k in range(N):
        if k > 0:
            hist = g[1:k + 1] @ e[k - 1::-1]
            e[k] = (w[k] - c * hist) / den
        q1, q2 = Q1[k], Q2[k]
        for _ in range(nsub):
            conv1 = U * A * (TaK - TH1); rad1 = eps * sig * A * (TaK**4 - TH1**4)
            conv2 = U * A * (TaK - TH2); rad2 = eps * sig * A * (TaK**4 - TH2**4)
            c12 = Us * As * (TH2 - TH1); r12 = eps * sig * As * (TH2**4 - TH1**4)
            dT1 = (conv1 + rad1 + c12 + r12 + a1 * q1) / mcp
            dT2 = (conv2 + rad2 - c12 - r12 + a2 * q2) / mcp
            TH1 += h * dT1; TH2 += h * dT2
        if k + 1 < N:
            w[k + 1] = [TH1 - TaK, TH2 - TaK]
    T = TaC + e
    return T[:, 0], T[:, 1]
def delayed(u, theta, ts=TS):
    d = theta / ts; n = int(np.floor(d)); f = d - n; N = len(u)
    u0 = np.zeros(N); u1 = np.zeros(N)
    if n < N: u0[n:] = u[:N - n]
    if n + 1 < N: u1[n + 1:] = u[:N - n - 1]
    return (1 - f) * u0 + f * u1

def sim_fopdt(p, u, ts=TS):
    K, tau, th = p; ud = delayed(u, th); a = np.exp(-ts / tau)
    y = np.zeros(len(u))
    for k in range(1, len(u)):
        y[k] = a * y[k - 1] + K * (1 - a) * ud[k - 1]
    return y

def sim_sopdt(p, u, ts=TS):
    K, t1, t2, th = p; ud = delayed(u, th)
    a1, a2 = np.exp(-ts / t1), np.exp(-ts / t2)
    x = np.zeros(len(u)); y = np.zeros(len(u))
    for k in range(1, len(u)):
        x[k] = a1 * x[k - 1] + K * (1 - a1) * ud[k - 1]
        y[k] = a2 * y[k - 1] + (1 - a2) * x[k - 1]
    return y

def sim_ffopdt(p, u, ts=TS):
    """tau_a D^alpha y + y = K u(t - theta), GL implícito."""
    K, tau_a, al, th = p; ud = delayed(u, th); N = len(u)
    g = gl_weights(al, N); c = tau_a * ts ** (-al); den = c + 1.0
    y = np.zeros(N)
    for k in range(1, N):
        y[k] = (K * ud[k - 1] - c * (g[1:k + 1] @ y[k - 1::-1])) / den
    return y

def sim_fsopdt(p, u, ts=TS):
    """K/((tau1 s + 1)(tau_b s^beta + 1)) e^{-theta s}: disipador entero + elemento fraccionario (sensor/difusión)."""
    K, t1, tb, be, th = p; ud = delayed(u, th); N = len(u); a1 = np.exp(-ts / t1)
    g = gl_weights(be, N); c = tb * ts ** (-be); den = c + 1.0
    x = np.zeros(N); y = np.zeros(N)
    for k in range(1, N):
        x[k] = a1 * x[k - 1] + K * (1 - a1) * ud[k - 1]
        y[k] = (x[k - 1] - c * (g[1:k + 1] @ y[k - 1::-1])) / den
    return y

MODELS = {
    'FOPDT':  dict(fn=sim_fopdt,  names=['K', 'tau', 'theta'],
                   lb=[1e-3, 1, 0], ub=[5, 3000, 300]),
    'SOPDT':  dict(fn=sim_sopdt,  names=['K', 'tau1', 'tau2', 'theta'],
                   lb=[1e-3, 1, 0.5, 0], ub=[5, 3000, 1000, 300]),
    'FFOPDT': dict(fn=sim_ffopdt, names=['K', 'tau_a', 'alpha', 'theta'],
                   lb=[1e-3, 0.5, 0.3, 0], ub=[5, 1e5, 1.9, 300]),
    'FSOPDT': dict(fn=sim_fsopdt, names=['K', 'tau1', 'tau_b', 'beta', 'theta'],
                   lb=[1e-3, 1, 0.5, 0.3, 0], ub=[5, 3000, 1e4, 1.5, 300]),
}

def two_point(t, u, y, t_step):
    """FOPDT por dos puntos 35,3 % / 85,3 % (Sundaresan-Krishnaswamy)."""
    m = t >= t_step; tt, yy = t[m] - t_step, y[m]
    u_pre = u[~m][-1] if (~m).any() else 0.0
    du = u[m][-1] - u_pre; dy = yy[-30:].mean() - yy[:5].mean()
    r = (yy - yy[:5].mean()) / dy
    t1 = tt[np.argmax(r >= 0.353)]; t2 = tt[np.argmax(r >= 0.853)]
    return dict(K=dy / du, tau=0.67 * (t2 - t1), theta=max(1.3 * t1 - 0.29 * t2, 0.0))

def starts(name, g2, rng):
    K, tau, th = g2['K'], max(g2['tau'], 5), g2['theta']
    base = {'FOPDT': [K, tau, th], 'SOPDT': [K, 0.7 * tau, 0.3 * tau, 0.5 * th],
            'FFOPDT': [K, tau, 1.0, th], 'FSOPDT': [K, 0.7 * tau, 0.3 * tau, 1.0, 0.5 * th]}[name]
    out = [np.array(base, float)]
    for _ in range(N_STARTS - 1):
        p = np.array(base) * rng.uniform(0.6, 1.5, len(base))
        if name == 'FFOPDT':
            p[2] = rng.uniform(0.6, 1.5); p[1] = tau ** p[2]
        if name == 'FSOPDT':
            p[3] = rng.uniform(0.6, 1.2); p[2] = (0.3 * tau) ** p[3]
        out.append(p)
    lb, ub = np.array(MODELS[name]['lb']), np.array(MODELS[name]['ub'])
    return [np.clip(p, lb + 1e-6, ub - 1e-6) for p in out]

def fit(name, u, y, idx, g2, seed=0):
    M = MODELS[name]; fn = M['fn']; n = idx.stop
    rng = np.random.default_rng(seed)
    res = lambda p: fn(p, u[:n])[idx] - y[idx]
    best = None
    for p0 in starts(name, g2, rng):
        r = least_squares(res, p0, bounds=(M['lb'], M['ub']), x_scale='jac', max_nfev=400)
        if best is None or r.cost < best.cost: best = r
    J = best.jac; npar = len(best.x); dof = max(len(best.fun) - npar, 1)
    s2 = 2 * best.cost / dof
    try: se = np.sqrt(np.diag(s2 * np.linalg.pinv(J.T @ J)))
    except Exception: se = np.full(npar, np.nan)
    return best.x, se

def metrics(yhat, y, idx_id, idx_val, npar):
    e_id, e_val = yhat[idx_id] - y[idx_id], yhat[idx_val] - y[idx_val]
    n = len(e_id); sse = float(e_id @ e_id)
    return dict(RMSE_id=np.sqrt(sse / n), RMSE_val=np.sqrt(np.mean(e_val**2)),
                AIC=n * np.log(sse / n) + 2 * npar, BIC=n * np.log(sse / n) + npar * np.log(n))
