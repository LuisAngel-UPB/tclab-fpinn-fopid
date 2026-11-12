# Closed-loop runs on the TCLab (NB04): representative repetition, the three operating points.
# Usage: python fig_nb04_kit.py <NB04_out/kit dir> <out dir> [rep]
import sys, pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
K, OUT = sys.argv[1], sys.argv[2]; REP = int(sys.argv[3]) if len(sys.argv) > 3 else 1
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times', 'Nimbus Roman', 'DejaVu Serif'], 'font.size': 8,
  'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'lines.linewidth': 0.8,
  'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'legend.frameon': False})
plan = pd.read_csv(f'{K}/plan_NB04.csv')
sty = {'P': ('#c0392b', 1.1, 'P (proposed)'), 'B1': ('#e08a1e', 0.8, 'B1'), 'B3': ('#3a9a5b', 0.8, 'B3'), 'B4': ('k', 0.8, 'B4')}
OPS = [35.0, 50.0, 65.0]
fig, ax = plt.subplots(2, 3, figsize=(7.16, 3.4), sharex=True, gridspec_kw=dict(height_ratios=[1.5, 1]))
for cid, (c, lw, lab) in sty.items():
    f = plan[(plan.id == cid) & (plan.rep == REP)].archivo.iloc[0]; D = pd.read_csv(f'{K}/corridas/{f}')
    for j, T in enumerate(OPS):
        d = D[(D.fase == 'ensayo') & (D.op == T)]; t = d.tau.values / 60
        if cid == 'P': ax[0, j].plot(t, d.r, color='0.55', ls='--', lw=0.8, label='$r$')
        ax[0, j].plot(t, d.T1, color=c, lw=lw, label=lab, zorder=3 if cid == 'P' else 2)
        ax[1, j].plot(t, d.Q1, color=c, lw=0.5 if cid != 'B1' else 0.8, alpha=0.9 if cid != 'P' else 1)
for j, T in enumerate(OPS):
    for a in ax[:, j]: a.axvline(15, color='0.75', lw=0.6)
    ax[0, j].set_title(rf'$\bar T = {T:.0f}$ °C', fontsize=8); ax[1, j].set_xlabel('Time (min)')
    ax[0, j].set_ylim(T - 0.7, T + 6.8); ax[1, j].set_xlim(0, 30)
    ax[0, j].text(15.3, T + 6.7, '$Q_2$: 0→40 %', fontsize=6.5, color='0.4', va='top')
ax[0, 0].set_ylabel('$T_1$ (°C)'); ax[1, 0].set_ylabel('$Q_1$ (%)')
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, ncol=5, fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, 1.0), handlelength=1.6, columnspacing=1.2)
fig.tight_layout(pad=0.3, w_pad=0.6, h_pad=0.4, rect=(0, 0, 1, 0.93))
fig.savefig(f'{OUT}/fig_exp_cl_kit.pdf'); fig.savefig(f'{OUT}/fig_exp_cl_kit.png', dpi=200)
