import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
O = sys.argv[1]
plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#52514e',
                     'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.grid': True,
                     'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
C1, C2 = '#2a78d6', '#eb6834'
# Fig 1: residuos del modelo físico entero -> modo común (ambiente)
fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=False)
for a, e in zip(axs, 'ABC'):
    R = pd.read_csv(f'{O}/residuos_fisico_{e}.csv')
    a.plot(R.time, R.r1, color=C1, lw=1.2, label='T1 medido − modelo')
    a.plot(R.time, R.r2, color=C2, lw=1.2, label='T2 medido − modelo')
    a.axhline(0, color='#52514e', lw=0.6)
    c = np.corrcoef(R.r1, R.r2)[0, 1]
    a.set_title(f'Experimento {e}  (correlación T1–T2 de los residuos = {c:.2f})', fontsize=9, loc='left')
    a.set_ylabel('residuo [°C]')
axs[2].legend(frameon=False, ncol=2, loc="lower right"); axs[-1].set_xlabel('t [s]')
fig.suptitle('Residuos del modelo físico entero (ajustado con A+B): T1 y T2 comparten derivas lentas (ambiente de la sala)', fontsize=9)
plt.tight_layout(); plt.savefig(f'{O}/fig_residuos_ambiente.png', dpi=200); plt.savefig(f'{O}/fig_residuos_ambiente.pdf')
# Fig 2: validación cruzada lineales
L = pd.read_csv(f'{O}/cv_lineales.csv')
mods = ['FOPDT', 'SOPDT', 'FFOPDT', 'FSOPDT']; cols = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
cases = [('G11', 'A'), ('G11', 'C'), ('G21', 'A'), ('G21', 'C')]
fig, a = plt.subplots(figsize=(7.2, 3.0)); w = 0.19
for j, (m, col) in enumerate(zip(mods, cols)):
    v = [L[(L.canal == ch) & (L.id == i) & (L.modelo == m)].RMSE_val.iloc[0] for ch, i in cases]
    a.bar(np.arange(4) + (j - 1.5) * w, v, w * 0.9, color=col, label=m)
    for k, (ch, i) in enumerate(cases):
        if m == 'FSOPDT':
            b = L[(L.canal == ch) & (L.id == i) & (L.modelo == m)].beta.iloc[0]
            a.text(k + 1.5 * w, v[k] + 0.02, f'β={b:.2f}', ha='center', fontsize=7, color='#52514e')
a.set_xticks(range(4)); a.set_xticklabels([f"{ch}\najuste {i} → valida {'C' if i == 'A' else 'A'}" for ch, i in cases], fontsize=7.5)
a.set_ylabel('RMSE de validación [°C]'); a.legend(frameon=False, ncol=4, fontsize=7.5, loc='upper right'); a.grid(axis='x', visible=False)
plt.tight_layout(); plt.savefig(f'{O}/fig_validacion_cruzada.png', dpi=200); plt.savefig(f'{O}/fig_validacion_cruzada.pdf')
