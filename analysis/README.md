# Analysis scripts

All scripts take input and output folders as command-line arguments. Run them from the
repository root; `out/` is any folder you choose.

## NB01_kit — identification checks (Sec. 6.1)

| script | what it does | outputs (archived here) |
|---|---|---|
| `scripts/lib.py` | shared model definitions (FOPDT, SOPDT, FFOPDT, grey-box energy balance) | – |
| `scripts/cv.py` | cross-validation of the linear models across experiments A, B, C | `cv_lineales.csv` |
| `scripts/phys2.py` | cross-validation of the physical grey-box model and residuals | `cv_fisico.csv`, `residuos_fisico_{A,B,C}.csv` |
| `scripts/figs.py` | residual vs. ambient drift and cross-validation figures | `fig_residuos_ambiente.pdf`, `fig_validacion_cruzada.pdf` |
| `scripts/q.py` | quick summary of the raw experiments | (console) |

```bash
D=data/NB01_kit_20261106/NB01_out; S=analysis/NB01_kit/scripts
python $S/cv.py    $D $S out/
python $S/phys2.py $D $S out/
python $S/figs.py  out/
```

## NB04_kit — closed-loop statistics (Sec. 6.3–6.5)

| script | what it does | outputs (archived here) |
|---|---|---|
| `scripts/win.py` | metrics over a fixed 300-s window after each step and paired tests | `metricas_ventana_300s.csv`, `resumen_ventana_300s.csv`, `pruebas_pareadas_ventana_300s.csv` |
| `scripts/m4_stats.py` | paired differences with 95 % CI and Holm correction; log-scale mixed model (operating point × controller) | `statistics/` |
| `scripts/m5_quant.py` | simulation of the 0.032 °C sensor quantization on the identified model | `quantization/` |
| `scripts/m6_noise.py` | noise amplification of each controller (|CS| and worst case) | `noise/` |
| `scripts/fig_nb04_kit.py` | closed-loop figure of the article (first repetition) | `fig_exp_cl_kit.pdf` |

```bash
K=data/NB04_kit_20261112/NB04_out/kit; C=data/NB03_kit_20261110/NB03_out/kit
N=data/NB02_kit_20261109_ABC/NB02_out/kit; S=analysis/NB04_kit/scripts
python $S/win.py        $K $C $N out/
python $S/m4_stats.py   $K out/
python $S/m5_quant.py   $C/controladores_NB04.json $N/surrogates/ABC_const_s0.json out/
python $S/m6_noise.py   $C/controladores_NB04.json $N/surrogates/ABC_const_s0.json out/
python $S/fig_nb04_kit.py $K out/ 1
```
