---
marp: true
theme: default
class: invert
backgroundColor: black
color: white
math: katex
---

## Session 4 — Glacier du Mont Miné: modelling deglaciation dates

Guillaume Jouvet

*Taught within the course "Dates and Rates of landscape evolution" (G. King and C. Schmidt)*

---

# Today (Session 4)

| | |
|---|---|
| Hour 1 | Modelling the Mont Miné since the Little Ice Age (this guide) |
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

# The folder: `session4/mont-mine/`

```
mont-mine/                       ← run igm_run from HERE
├── analysis.ipynb               notebook: plots only (maps, sample sites, deglaciation years)
├── Samples_2024.csv             sample sites (ID, CH1903 x/y, UTM32 x/y)
├── Samples_XXXX.csv             a second set of sample sites (19FER...)
├── experiment/
│   ├── params.yaml              1700-2024, ELA-based SMB
│   └── common/iceflow_offline.yaml   ice-flow emulator (do not touch)
└── outputs/, data/              created by IGM
```

Same tools as before: **YAML** (parameters) + **terminal** (runs) + **notebook** (plots)

---

# The model set-up (`experiment/params.yaml`)

- `oggm_shop` downloads the data of the glacier (topography, ice thickness, outline): `RGI2000-v7.0-G-11-00907`
- Run from **1700 to 2024**, starting from **today's geometry**
- SMB from the ELA (`smb: method: simple`): ELA **lowered** first (advance to the LIA maximum, ~1850), then **raised** (retreat until today)

$$SMB(z) = \beta\,(z - z_{ELA}), \quad \beta = \texttt{gradabl} \text{ below},\ \texttt{gradacc} \text{ above, capped at } \texttt{accmax}$$

- Ice flow: emulator `dahunet_edu`, `tau_ref`, `arrhenius` (all visible in `params.yaml`)
- Grid: `coarsening: ratio: 2` for fast tests, **`ratio: 1` for your final runs**

---

# Step 1 — Set up and run

1. Open `session4/mont-mine/analysis.ipynb`, run the first cell
2. *File → New → Terminal*, next to the notebook:

```bash
cd ~/work/glamod/session4/mont-mine
igm_run +experiment=params hydra.run.dir=outputs/run_v1
```

- The first run downloads the data (≈ 1 min), then runs 1700–2024 (a few minutes)
- Notebook, section 1: the SMB profile, the dashboard, volume and area through time

---

# Step 2 — Ice cover at the sample sites

Notebook, section 2:

- `plot_samples_map`: today's glacier and the sample sites, and the modelled LIA glacier (1850)
- `animate_glacier_evolution`: the retreat since 1700
- `plot_thk_at_samples`: ice thickness through time at each site, and the **modelled deglaciation year** (last time the site is covered by ice)

Enter the **measured ages** in `ages` (calendar years, or years before 2024) to compare them with the model

---

# Step 3 — Calibration: your turn!

Copy `experiment/params.yaml` (e.g. `params_v2.yaml`), change **one thing at a time**, run into a new folder:

```bash
igm_run +experiment=params_v2 hydra.run.dir=outputs/run_v2
```

1. **ELA history**: when and how low is the LIA maximum? How fast does the ELA rise afterwards?
2. **SMB gradients** and maximum accumulation (`gradabl`, `gradacc`, `accmax`)
3. **Basal sliding** `tau_ref`: e.g. 0.4 (less sliding) and 0.1 (more sliding)

Then compare the deglaciation years with the measured ages (notebook, section 3)

---

# Questions

- Which parameters control the **timing** of deglaciation at the sites, and which control the **extent** of the LIA glacier?
- Can you match all samples with a single ELA history? If not, why (model, data, sample interpretation)?
- How sensitive are the modelled dates to the grid resolution (`ratio: 2` vs `ratio: 1`)?

---

# To go further

- **Climate-driven SMB** instead of a prescribed ELA: `climate` + `smb` with `method: oggm`, as in the mini-project (`session2/my-glacier/experiment/params_past.yaml`). Observed climate from 1901; before 1901, the cooling is set with `delta_temp` in `clim_trend_array`
- **Second set of sample sites**: `Samples_XXXX.csv` (IDs `19FER...`), same format: `plot_thk_at_samples(..., "Samples_XXXX.csv")`
