---
marp: true
theme: default
class: invert
backgroundColor: black
color: white
math: katex
---

# Session 4 — Glacier du Mont Miné: modelling deglaciation dates

Guillaume Jouvet

*Taught within the course "Dates and Rates of landscape evolution" (G. King and C. Schmidt)*

---

# Today (Session 4)

| | |
|---|---|
| Hour 1 | Modelling the Mont Miné since the Little Ice Age (this guide), ice-flow calibration |
| Hour 2 | G. King's course |
| Hours 3–4 | Calibrate the model against the sample ages |

New to IGM? Go through Lecture 2 first (`session2/1-lecture-igm.md`).

---

# The question

[Glacier du Mont Miné](https://map.geo.admin.ch/#/map?lang=en&center=2607500,1100500&z=6) (Val d'Hérens, Valais): how did it retreat since the Little Ice Age (LIA, ~1850)?

- **Rock samples** collected in the glacier forefield are dated (OSL / exposure ages)
- For each sample site, the model gives the **year when the site became ice-free**
- Goal: **calibrate the glacier model** so that modelled and measured dates agree

Retreat of the Mont Miné since the LIA: slides 28–30 of [this presentation](https://www.unige.ch/forel/application/files/3116/5476/6042/F-Evolene-Paysagesglaciaires_compressed.pdf)

---

# The folder: `mont-mine/`

Copy it **outside** the course repository (`git pull` would otherwise conflict with your changes), then work there:

```bash
cp -r ~/work/glamod/session4/mont-mine ~/work/mont-mine
cd ~/work/mont-mine              # run igm_run from HERE
```

```
mont-mine/
├── analysis.ipynb               notebook: plots only (speeds, maps, sample sites, deglaciation years)
├── Samples_2024.csv             sample sites (ID, CH1903 x/y, UTM32 x/y): indicative, replace by YOURS
├── Samples_XXXX.csv             a second (indicative) set of sample sites (19FER...)
├── experiment/
│   ├── params_sensitivity.yaml  1-year runs from today: ice-flow calibration (Step 1)
│   └── params.yaml              1700-2024, ELA-based SMB (+ ice flow, emulator, run folder)
└── outputs/, data/              created by IGM (outputs/params/ = run of params.yaml)
```

---

# The model set-up (`experiment/params.yaml`)

- `oggm_shop` downloads the data of the glacier (topography, ice thickness, outline): `RGI2000-v7.0-G-11-00907`
- Run from **1700 to 2024**, starting from **today's geometry**
- SMB from the ELA (`smb: method: simple`): ELA **lowered** first (advance to the LIA maximum, ~1850), then **raised** (retreat until today)

$$SMB(z) = \beta\,(z - z_{ELA}), \quad \beta = \texttt{gradabl} \text{ below},\ \texttt{gradacc} \text{ above, capped at } \texttt{accmax}$$

- Ice flow: emulator `dahunet_edu`, `tau_ref`, `arrhenius` (all visible in `params.yaml`)
- Grid: `coarsening: ratio: 2` for fast tests, **`ratio: 1` for your final runs**

---

# Step 1 — Ice-flow calibration (as in Session 2)

1. Open `~/work/mont-mine/analysis.ipynb`, run the first cell
2. *File → New → Terminal*: grid search of `arrhenius` × `tau_ref`, 1-year runs from **today's** glacier (12 runs; the first one downloads the data):

```bash
cd ~/work/mont-mine
igm_run -m +experiment=params_sensitivity \
   processes.iceflow.physics.viscosity.arrhenius=25,50,100 \
   processes.iceflow.physics.sliding.tau_ref=0.1,0.2,0.4,0.8
```

3. Notebook, section 1: modelled − observed speeds for each couple ➡️ pick the couple with the **smallest misfit** (bias ≈ 0, small RMSE), refine around it if needed
4. `experiment/params.yaml` uses the best couple of this grid (`arrhenius` = 100, `tau_ref` = 0.4): if you find a better one by refining, put it in its `iceflow → physics` block

---

# Step 2 — The reference run

```bash
igm_run +experiment=params
```

- 1700–2024 (a few minutes) ➡️ `outputs/params/`
- Notebook, section 2: the SMB profile, the dashboard, volume and area through time

---

# Step 3 — Ice cover at YOUR sample sites

The samples provided (`Samples_2024.csv`) are **only indicative**, to get things running: **replace them by your own samples** (same format: ID, CH1903 x/y, UTM32 x/y), and set `SAMPLES` in the first cell of the notebook

Notebook, section 3:

- `plot_samples_map`: today's glacier and the sample sites, and the modelled LIA glacier (1850)
- `animate_glacier_evolution`: the retreat since 1700
- `plot_thk_at_samples`: ice thickness through time at each site, and the **modelled deglaciation year** (last time the site is covered by ice)

Enter **your measured ages** in `ages` (calendar years, or years before 2024)

---

# Step 4 — Calibration of the ELA history: your turn!

Copy `experiment/params.yaml` to `experiment/my.params.yaml`, change **only the ELA history** (`ela` column of the `smb` array; keep the ice flow of Step 1 and the SMB gradients), run it (➡️ `outputs/my.params/`):

```bash
igm_run +experiment=my.params
```

- When and how low is the LIA maximum? How fast does the ELA rise afterwards?
- Change one thing at a time, until the modelled deglaciation years match **your** sample ages (notebook, section 4)

---

# Questions

- Which part of the ELA history controls the **timing** of deglaciation at the sites, and which controls the **extent** of the LIA glacier?
- Can you match all samples with a single ELA history? If not, why (model, data, sample interpretation)?
- How sensitive are the modelled dates to the grid resolution (`ratio: 2` vs `ratio: 1`)?

---

# To go further

- **Climate-driven SMB** instead of a prescribed ELA: `climate` + `smb` with `method: oggm`, as in the mini-project (`session2/my-glacier/experiment/params_past.yaml`). Observed climate from 1901; before 1901, the cooling is set with `delta_temp` in `clim_trend_array`
- **Second set of sample sites**: `Samples_XXXX.csv` (IDs `19FER...`, indicative), same format: `plot_thk_at_samples(..., "Samples_XXXX.csv")`
- **Other parameters**: SMB gradients and maximum accumulation (`gradabl`, `gradacc`, `accmax`): how sensitive are the deglaciation dates?
