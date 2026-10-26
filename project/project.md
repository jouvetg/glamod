# Mini-project: model your own glacier

## Goal

Each student models **one Swiss glacier** (chosen in Session 1, see [choose your glacier](../session1/1e-choose-your-glacier.md)) with IGM, to answer:

> **How did my glacier evolve since the Little Ice Age (LIA, ~1850), and how will it evolve until 2100 under different climate scenarios?**

You work in the folder `project/my-glacier/` (experiment files, `run_future.sh` and the analysis notebook `analysis.ipynb`), during Sessions 2 and 3 (≈ 6 hours in class, with the help of the assistant), and write a short report.

## Timeline

| When | What |
|---|---|
| Session 1 (last hour) | Choose your glacier, collect information and observations |
| Session 2 (hours 3–4) | **Present**: download data, check the initial state. **Past**: first LIA → today runs |
| Session 3 (hours 2–4) | Finish the past calibration. **Future**: scenario runs. Figures, report outline |
| After Session 3 | Write the report — deadline: **to be announced** |

## The template: `project/my-glacier/`

```
my-glacier/
├── analysis.ipynb                # notebook: plot, compare, make figures — your lab book
├── run_future.sh                 # runs the future scenarios one after the other
├── figures/                      # (created by the notebook) figures for the report
├── outputs/                      # (created by IGM) one folder per run: outputs/past_v1/, outputs/future_rcp45/, ...
└── experiment/
    ├── common/
    │   ├── my_glacier.yaml       # YOUR RGI ID + ice-flow parameters (shared by all experiments)
    │   └── iceflow_offline.yaml  # ice-flow emulator settings (do not touch)
    ├── params_past.yaml          # 1700-2020, climate-driven SMB
    ├── params_past_simple.yaml   # 1700-2020, ELA-based SMB (simpler alternative)
    └── params_future.yaml        # 2020-2100, climate-driven SMB, one scenario
```

**First thing to do**: put the RGI ID of your glacier in `experiment/common/my_glacier.yaml`.

## How to work: YAML + terminal + notebook

Open the three side by side in JupyterLab (drag the tabs to split the screen):

| What | Where | Good practice |
|---|---|---|
| **Set** the parameters | YAML files in `experiment/`, in the text editor | One file = one experiment: copy a file before changing it (right-click → *Duplicate*, e.g. `params_past_v2.yaml`) |
| **Run** IGM | **Terminal**, from `~/work/glamod/project/my-glacier` | Always name the run folder: `igm_run +experiment=params_past_v2 hydra.run.dir=outputs/past_v2` |
| **Look** at the results | `analysis.ipynb` | Postprocessing only: it reads `outputs/<run>/` and saves figures to `figures/` |

A run in the terminal keeps going if you close the browser tab, and the notebook stays free for plotting while IGM runs. Each run folder keeps all the parameters used (`.hydra/config.yaml`), so every result can be traced back. Keep a short log (in the notebook) of each run: what you changed, why, and what came out.

## Step 1 — Present: check the initial state

- Run any experiment once (the notebook gives a 1-year run): the data is downloaded to `data/` (`input_<RGI_ID>.nc` and `data/<RGI_ID>/`).
- Plot the input data (`plot_input_data`) and the climate (`plot_climate`): area, volume, thickness, observed speeds.
- Compare the modelled surface speed at the start of a run with the observed one (`compare_modeled_observed_velocity`). If the model is systematically too fast or too slow, adjust the sliding (`tau_ref`) or the ice softness (`arrhenius`) in `common/my_glacier.yaml`.
- Compare with what you know: GLAMOS length and volume changes, the swisstopo thickness map.

## Step 2 — Past: from the Little Ice Age to today

We do not know the geometry of the glacier in the past. The usual trick: **start in 1700 from today's geometry**, cool the climate so that the glacier advances to its LIA maximum (~1850), then let it retreat with the observed climate of 1901–2019. If the model reproduces both the **LIA extent** (moraines, Siegfried map) and the **present-day extent**, we gain confidence in it before making projections.

Two options:

1. **`params_past.yaml`** (recommended) — climate-driven SMB (`climate` + `smb`, method `oggm`):
   - 1901–2019: the observed monthly climate is used.
   - Before 1901: random years from 1960–1990, shifted by `delta_temp` (°C, relative to 1960–1990) and scaled by `prec_scal`, as set in `clim_trend_array`.
   - Calibrate the **pre-1901 `delta_temp`** (LIA maximum) and **`melt_enhancer`** (present-day extent).
2. **`params_past_simple.yaml`** — ELA-based SMB: you prescribe the ELA history yourself. Simpler to understand, but less realistic. Estimate today's ELA from the end-of-summer snowline or GLAMOS. LIA ELAs were typically 100–200 m lower than in the late 20th century.

**Mind the response time**: the glacier lags behind the climate. Large glaciers respond in a century or more, so the 1700–1850 spin-up must be long enough.

## Step 3 — Future: 2020–2100

`params_future.yaml` starts from today's glacier and uses the climate after 2019: random years from 1960–1990, plus `delta_temp`. Since `delta_temp` is **relative to 1960–1990**, compute it as follows:

- **2020 row**: today's offset vs. 1960–1990. `plot_climate(RGI_ID)` prints it (≈ +1.3 °C for 2000–2019 at Findelen).
- **2100 row**: offset of 1981–2010 vs. 1960–1990 (also printed by `plot_climate`) **+** the scenario warming relative to 1981–2010.

Scenario warming for Switzerland around 2085 vs. 1981–2010 (annual mean, [CH2018](https://www.nccs.admin.ch/nccs/en/home/climate-change-and-impacts/swiss-climate-change-scenarios.html)):

| Scenario | Warming vs. 1981–2010 |
|---|---|
| RCP2.6 / SSP1-2.6 (strong mitigation) | ≈ +0.6 to +1.9 °C |
| RCP4.5 / SSP2-4.5 (moderate) | ≈ +2.0 to +3.3 °C |
| RCP8.5 / SSP5-8.5 (high emissions) | ≈ +3.3 to +5.4 °C |

`run_future.sh` runs four scenarios (committed, RCP2.6, RCP4.5, RCP8.5) with these rules: set your calibrated `melt_enhancer` and the two offsets at the top of the file, then `bash run_future.sh` in the terminal. You can change the warming values or add scenarios (e.g. the low and high ends of the CH2018 ranges). Run **at least two scenarios**, using the `melt_enhancer` calibrated in Step 2. Analyse volume and area loss, front retreat, and when the glacier (nearly) disappears. Add a **committed-loss** experiment: keep today's climate until 2100 (2100 row = 2020 row). How much would still be lost?

## Step 4 — Report

See the guidelines in [Lecture 3](../session3/lecture-3-application-and-report.md). In short:

- **≤ 1500 words** (excluding captions and references), PDF.
- Structure: introduction and research question(s), study site, methods (describe **what** the model does, not how you ran it: no code, no YAML), results, discussion (limitations, uncertainties), conclusion.
- 4–6 good figures: maps, time series of volume/area, flowline profiles, model vs. observations, scenarios.
- Evaluation: (A) research question, context and literature; (B) modelling tools and experiments; (C) results and discussion; (D) form and quality; on-time submission.

## Calibration tips

| Problem | Try |
|---|---|
| The glacier grows without limit | Climate too cold / ELA too low → increase `delta_temp` or `melt_enhancer` (or raise the ELA) |
| The glacier disappears too fast | Climate too warm / ELA too high → the opposite |
| No LIA advance | Cooler pre-1901 climate (more negative `delta_temp`), or a longer spin-up |
| Speeds too high vs. observations | Increase `tau_ref` (less sliding), or decrease `arrhenius` (harder ice) |
| Speeds too low | The opposite |

- Change **one parameter at a time**, and keep track of all your runs.
- Do not expect a perfect fit: aim at the right trends and orders of magnitude, and discuss the mismatch.
- To test quickly, shorten runs from the command line (e.g. `processes.time.end=1800`).

## Going further (optional)

- **Moraines**: track debris with the `particles` module (see igm-examples `aletsch`, Part A, Step 4).
- **Sensitivity**: how do the 2100 results depend on `tau_ref`, `arrhenius`, the initial thickness (`thk_source: consensus_ice_thickness`), or precipitation (`prec_scal`)?
- **Response time**: apply a step change in climate and measure how long the glacier takes to adjust.
- **Your own question**: proglacial lakes, tributary disconnection, hazards, ... Discuss it with us first.

## Practicalities

- Work in your RenkuLab session ([setup/renku.md](../setup/renku.md)). Copy your notebook, YAML files, `run_future.sh` and figures to your **SWITCHdrive folder** regularly: sessions are not permanent storage.
- Data for your glacier is downloaded once, then re-used. If something looks wrong with the data, delete `data/input_<RGI_ID>.nc` and run again.
