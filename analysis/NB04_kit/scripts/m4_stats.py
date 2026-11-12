"""M4 (revisión CEP): IC 95 %, Holm y modelo mixto sobre las 15 corridas de NB04."""
import sys, numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
base = sys.argv[1]; out = sys.argv[2]
run = pd.read_csv(f"{base}/metricas_por_corrida.csv")
tr = pd.read_csv(f"{base}/metricas_por_tramo.csv")
METS = ["IAE_r", "IAE_d", "J_normD", "TV"]
COMPS = ["B1", "B2", "B3", "B4"]
rows = []
for m in METS:
    for c in COMPS:
        p = run[run.id == "P"].set_index("rep")[m]; b = run[run.id == c].set_index("rep")[m]
        d = (p - b).dropna(); n = len(d)
        md, se = d.mean(), d.std(ddof=1) / np.sqrt(n); tq = stats.t.ppf(0.975, n - 1)
        pv = stats.ttest_rel(p.loc[d.index], b.loc[d.index]).pvalue
        bm = b.loc[d.index].mean()
        # log-ratio per block (multiplicative effect)
        lr = np.log(p.loc[d.index] / b.loc[d.index]); lm, lse = lr.mean(), lr.std(ddof=1) / np.sqrt(n)
        rows.append(dict(metrica=m, comparador=c, n=n, P=p.mean(), comp=bm, dif=md,
                         dif_ci_lo=md - tq * se, dif_ci_hi=md + tq * se,
                         cambio_pct=100 * md / bm, cambio_pct_lo=100 * (md - tq * se) / bm, cambio_pct_hi=100 * (md + tq * se) / bm,
                         ratio_pct=100 * (np.exp(lm) - 1), ratio_lo=100 * (np.exp(lm - tq * lse) - 1), ratio_hi=100 * (np.exp(lm + tq * lse) - 1),
                         p_t=pv))
df = pd.DataFrame(rows)
df["p_holm"] = np.nan
for m in METS:
    i = df.metrica == m
    df.loc[i, "p_holm"] = multipletests(df.loc[i, "p_t"], method="holm")[1]
df.to_csv(f"{out}/pareadas_IC_holm.csv", index=False)

# Mixed model: log(metric) ~ controller + op, random intercept per block (rep); 45 tests
mm = []
tr = tr.copy(); tr["ctrl"] = pd.Categorical(tr.id, ["P"] + COMPS); tr["op"] = tr.op.astype(int).astype(str)
for m in ["IAE_r", "IAE_d"]:
    d = tr[tr[m] > 0].copy(); d["y"] = np.log(d[m])
    for formula in ["y ~ C(ctrl, Treatment('P')) + op"]:
        fit = smf.mixedlm(formula, d, groups=d["rep"]).fit(reml=True)
        ci = fit.conf_int()
        for c in COMPS:
            k = f"C(ctrl, Treatment('P'))[T.{c}]"
            # coef = log(comp/P) -> P vs comp = exp(-coef)-1
            mm.append(dict(metrica=m, comparador=c, P_vs_comp_pct=100 * (np.exp(-fit.params[k]) - 1),
                           lo=100 * (np.exp(-ci.loc[k, 1]) - 1), hi=100 * (np.exp(-ci.loc[k, 0]) - 1), p=fit.pvalues[k]))
mmd = pd.DataFrame(mm)
mmd["p_holm"] = np.nan
for m in mmd.metrica.unique():
    i = mmd.metrica == m; mmd.loc[i, "p_holm"] = multipletests(mmd.loc[i, "p"], method="holm")[1]
mmd.to_csv(f"{out}/modelo_mixto_log.csv", index=False)

# P vs comp por punto de operación (media de 3 rep), IAE_d e IAE_r
po = tr.groupby(["id", "op"])[["IAE_r", "IAE_d"]].mean().unstack("id")
po.to_csv(f"{out}/medias_por_op.csv")
pd.set_option("display.width", 200); pd.set_option("display.precision", 3)
print(df[["metrica","comparador","cambio_pct","cambio_pct_lo","cambio_pct_hi","p_t","p_holm"]])
print(mmd); print(po)
