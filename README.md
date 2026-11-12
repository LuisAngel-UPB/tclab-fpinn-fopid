# Gradient-based tuning of FOPID controllers through physics-informed fractional surrogates — data and code

[![DOI](https://zenodo.org/badge/1455559155.svg)](https://doi.org/10.5281/zenodo.24168802)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)

This repository contains the Jupyter notebooks, the experimental data and the analysis
scripts that support the article

> L. Angel, *Gradient-based tuning of FOPID controllers through physics-informed
> fractional surrogates: experimental validation on a thermal process*,
> submitted to *Control Engineering Practice* (2026).

The workflow has four stages, one notebook each:

| Notebook | Stage | Hardware | Manuscript |
|---|---|---|---|
| `NB01_identificacion_TCLab.ipynb` | Open-loop experiments A, B, C on the TCLab and classical identification (FOPDT/SOPDT, FFOPDT, grey-box energy balance) | TCLab kit | Sec. 5.2, Sec. 6.1 (Fig. `fig:exp_id`, Table `tab:exp_id`) |
| `NB02_fPINN_kit.ipynb` | Fractional physics-informed neural network (fPINN) identification of the TCLab model and fractional orders; integer-order PINN and output-error baselines | none (CPU) | Sec. 6.1 (Fig. `fig:fpinn_val`, Table `tab:exp_par`) |
| `NB02_fPINN_simulacion.ipynb` | Same identification on a simulated plant with known orders (α\*, β\*) | none (CPU) | Sec. 4.3 (Table `tab:sim_id`) |
| `NB03_sintonia_FOPID.ipynb` | Differentiable FOPID tuning (Kp, Ki, Kd, λ, μ, Kt) through the frozen surrogate with Ms/PM penalties; baselines B1–B4 and ablations A1–A2 | none (CPU) | Sec. 6.2 (Table `tab:exp_tuning`, Fig. `fig:tuning`), Sec. 4.4 |
| `NB03_sintonia_FOPID_designOE.ipynb` | Variant of NB03 that tunes on the integer-order output-error model instead of the fPINN (check requested in revision) | none (CPU) | Sec. 6.2 / Discussion |
| `NB04_lazo_cerrado_TCLab.ipynb` | Closed-loop tests: setpoint step + Q2 disturbance at 35/50/65 °C, 5 controllers × 3 repetitions, metrics and paired tests | TCLab kit (or `HW='sim'`) | Sec. 6.3–6.5 (Fig. `fig:exp_cl`, `fig:exp_ops`, Tables `tab:exp_cl`, `tab:exp_transfer`) |

Notebook comments and variable names are in Spanish; the article, this README and the
data dictionary are in English.

## Repository layout

```
notebooks/                 the six notebooks above (outputs cleared)
data/                      archived results exactly as produced on the author's PC
  NB01_kit_20261106/       raw open-loop data (expA/expB/expC) + identified models
  NB02_kit_20261109_splits/ fPINN results with train/validation splits AB→C and BC→A
  NB02_kit_20261109_ABC/   same plus the model trained on A+B+C (the surrogate used for tuning)
  NB03_kit_20261110/       tuned controllers (fPINN design, Tf = 10 s) used in NB04
  NB03_kit_20261112_designOE/ tuning on the output-error model (revision check)
  NB04_kit_20261112/       15 closed-loop runs on the kit, per-run/per-segment metrics, paired tests
analysis/                  post-processing scripts and their outputs
  NB01_kit/                cross-validation of the identified models, ambient-drift residuals
  NB04_kit/                300-s window metrics, Holm-corrected paired tests and mixed model,
                           quantization and noise simulations, figure script
requirements.txt  CITATION.cff  .zenodo.json  LICENSE (code, MIT)  LICENSE-DATA (data, CC BY 4.0)
```

## Hardware and software

* Temperature Control Lab (TCLab), Arduino Leonardo, firmware 2.0.1, `tclab` 1.0.0, sampling time 1 s,
  ambient ≈ 28 °C.
* Laptop with an AMD Ryzen 7 PRO 5850U processor (8 cores, 16 threads) and 16 GB of RAM, Windows 10,
  Python 3.9.13, PyTorch 2.8.0 (CPU). No GPU was used.

## How to reproduce

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook
```

Every notebook has a configuration cell at the top:

* `MODE = 'quick'` runs a short smoke test (minutes); `MODE = 'full'` reproduces the article
  (several hours for NB02 and NB03 on a laptop CPU; NB04 takes about 2 h of kit time per run, ≈ 38 h for the 15 runs).
* Results are written to `notebooks/NBxx_out/` and zipped (`NBxx_*_resultados_*.zip`); they are ignored by git.
* To start a later notebook from the **archived** results instead of re-running the previous one,
  point its input variable to `data/`:

| Notebook | Variable | Archived input |
|---|---|---|
| NB02_fPINN_kit | `DATA_DIR` | `'../data/NB01_kit_20261106/NB01_out'` |
| NB03 | `NB02_DIR` | `'../data/NB02_kit_20261109_ABC/NB02_out/kit'` |
| NB04 | `NB03_DIR`, `NB02_DIR` | `'../data/NB03_kit_20261110/NB03_out/kit'`, `'../data/NB02_kit_20261109_ABC/NB02_out/kit'` |

  If these variables are left as `None`, each notebook looks for `NBxx_out/` (or the newest
  `NBxx_*_resultados_*.zip`) in the current folder and its parent.
* NB01 and NB04 with `HW='kit'` need the TCLab connected over USB and must start from thermal
  equilibrium (kit off and cold). NB04 can be run without hardware with `HW='sim'`, which uses
  the identified model as a virtual plant.

The analysis scripts take their input and output folders as arguments, e.g. from the repository root:

```bash
python analysis/NB04_kit/scripts/m4_stats.py data/NB04_kit_20261112/NB04_out/kit  out/
python analysis/NB04_kit/scripts/win.py data/NB04_kit_20261112/NB04_out/kit \
       data/NB03_kit_20261110/NB03_out/kit data/NB02_kit_20261109_ABC/NB02_out/kit  out/
```

See `analysis/README.md` for the full list.

## Data dictionary (main files)

**Open-loop experiments** — `data/NB01_kit_20261106/NB01_out/exp{A,B,C}.csv`

| column | unit | description |
|---|---|---|
| `time` | s | time from the start of the experiment (nominal 1 s grid) |
| `Q1`, `Q2` | % | heater commands (0–100 %) |
| `T1`, `T2` | °C | measured temperatures |
| `t_real` | s | wall-clock time at which the sample was read |

A = steps in Q1, B = steps in Q2, C = multilevel excitation of both heaters (plan in `plan_expC.csv`).
`metadatos_kit.json` stores port, firmware, OS, ambient temperature and per-experiment timing.
`expA_cortado_*.csv` is an interrupted first attempt of experiment A kept for transparency; it is not used.

**Closed-loop runs** — `data/NB04_kit_20261112/NB04_out/kit/corridas/corrida_NN_<id>_rep<r>.csv`

| column | unit | description |
|---|---|---|
| `t` | s | time from the start of the run |
| `fase` | – | `aprox`: approach to the operating point; `ensayo`: test segment (setpoint step of +5 °C at `tau` = 0, Q2 step of 40 % at `tau` = 900 s) |
| `op` | °C | operating point (35, 50 or 65) |
| `tau` | s | time since the start of the current segment |
| `r` | °C | setpoint for T1 |
| `T1`, `T2` | °C | measured temperatures |
| `Q1`, `Q2` | % | applied heater commands (Q1 saturated to 0–100 %) |
| `v` | % | controller output before saturation (used by the anti-windup) |
| `t_real` | s | wall-clock time of the sample |

Controller ids: `P` proposed FOPID (fPINN surrogate, differentiable tuning), `B1` FOPID tuned on an FFOPDT model,
`B2` FOPID tuned by PSO on the same fPINN, `B3` PID tuned on the integer-order PINN, `B4` SIMC PID on SOPDT.
`plan_NB04.csv` lists the 15 runs used, in the order they were executed; `corrida_01_B4_rep1.csv` (incomplete first attempt)
and `corrida_09_B4_rep2_cortada_*.csv` (interrupted run, repeated as `corrida_09_B4_rep2.csv`) are kept for
transparency and are not used. Per-run and per-segment metrics are in `metricas_por_corrida.csv` and
`metricas_por_tramo.csv` (IAE, ITAE, TV, energy, overshoot, settling time, saturation).

**Identified surrogates** — `data/NB02_*/NB02_out/kit/surrogates/<split>_<ambient>_s<seed>.json`
hold the identified parameters (U, Us, α1, α2, τs, fractional orders α, β) for each split and seed;
`ABC_const_s0.json` is the surrogate used for tuning. `resultados_kit.csv` summarizes all models.

**Controllers** — `data/NB03_kit_20261110/NB03_out/kit/controladores.csv` (Kp, Ki, Kd, λ, μ, Kt and
tuning time for P, B1–B4, A1, A2); `controladores_NB04.json` is the file read by NB04.

## Citation

Please cite the article and this archive (see `CITATION.cff`; GitHub shows a "Cite this repository" button).
Zenodo DOI (all versions): [10.5281/zenodo.24168802](https://doi.org/10.5281/zenodo.24168802).

## Licence

Code (`*.ipynb`, `*.py`): MIT, see [LICENSE](LICENSE).
Data and figures: CC BY 4.0, see [LICENSE-DATA](LICENSE-DATA).

## Contact

Luis Angel — Universidad Pontificia Bolivariana, Bucaramanga, Colombia —
ORCID [0000-0002-9156-9547](https://orcid.org/0000-0002-9156-9547).
Questions and bug reports: please open an issue in this repository.
