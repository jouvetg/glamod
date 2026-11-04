---
marp: true
theme: default
class: invert
backgroundColor: black
color: white
math: katex
---

<style>
/* centre single images */
p > img:only-child:not(.emoji) { display: block; margin: 0 auto; }
/* slides tagged center-img: centre all paragraphs (e.g. two images side by side) */
section.center-img p { text-align: center; }
</style>

# Session 2 — Lecture 2: Introduction to IGM
  

---

# Course format: 3 afternoons × 4 hours + 1 extra

| | Hour 1 | Hour 2 | Hour 3 | Hour 4 |
|---|---|---|---|---|
| **Week 1** | Glacier processes | Glacier modelling | *Tutorial* | *Glacier selection* |
| **Week 2** | **Intro to IGM**, **past & future runs** | **Project** | **Project** | **Project** |
| **Week 3** | Extra experiments, report | *Project* | *Project* | *Project* |
| *(Week 4* | *dates and rate add-on* | *G. King's course* | *Project* | *Project)* |
 

---

# From the tutorial to IGM

Tutorial (Session 1): 1D/2D SIA minimal model, synthetic SMB, very simple code.

A "real" glacier model needs:
- **real data**: topography, ice thickness, outlines, velocities, climate
- a **better ice-flow model** than SIA (steep, narrow valleys; sliding)
- a **better SMB model** (driven by temperature and precipitation)
- **efficiency**: many runs to explore parameters (calibration, scenarios)

➡️ IGM: an open-source Python model developed @ UNIL, https://igm-model.org

![height:100px](fig/logo_igm.png)

---

# IGM in one picture: a 3D glacier evolution model

![width:1200px](fig/igm-overview.png)


---

# What does IGM solve?

Same structure as the tutorial, at each time step, on a raster grid:

1. **Surface mass balance**: $SMB(x,y,t)$ from an ELA or from climate
2. **Ice flow**: velocity field from a higher-order ice flow approximation (no SIA), 
using a convolutional neural network / image-to-image mapping:
![width:300px](fig/igm-cnn-mapping.png)
3. **Mass conservation**: $\dfrac{\partial h}{\partial t} + \nabla\cdot(\bar{\mathbf{u}}\,h) = SMB$.
4. Update time, write outputs
 
---

# IGM is modular

![width:1200px](fig/igm-flowchart.png)

---

# Structure of an IGM run folder

```
my-glacier/                      ← run igm_run from HERE
├── experiment/                  PARAMETERS
│   └── params_past.yaml             one YAML file = one experiment
├── data/                        INPUTS (downloaded by oggm_shop at the first run)
│   └── input_<RGI_ID>.nc            topography, thickness, velocities, climate...
└── outputs/                     OUTPUTS: one folder per run (hydra: run: dir: at the end of the file)
    └── params_past/
        ├── output.nc                maps through time
        ├── output_ts.nc             volume and area through time
        ├── dashboard_*.png          quick-look images
        └── .hydra/                  all parameters used (config.yaml)
```

---

<style scoped>pre { font-size: 0.62em; }</style>

# A parameter file (`experiment/params.yaml`)

```yaml
defaults:
  - override /inputs: [local]                        # 1) which data
  - override /processes: [smb, iceflow, time, thk]   # 2) which physics, in order
  - override /outputs: [local, live_dashboard]       # 3) which outputs

inputs:
  local:
    coarsening:
      ratio: 2                       # GRID: 2 = fast tests, 1 = final runs
processes:                           # only what differs from the defaults
  smb:
    method: simple                   # + the ELA history (array)
  iceflow:                           # ICE FLOW
    physics:
      sliding:
        tau_ref: 0.325
    # + the emulator settings (set for you, do not modify)
  time:
    start: 1900.0
    end: 2000.0
    save: 10.0
```

Run it: `igm_run +experiment=params`

---

# Changing parameters

- Change parameters **in the YAML file**, and **keep 1 file per experiment**: copy it before changing it (`params_past.yaml` → `params_past_v2.yaml`), so that every experiment stays documented
- Run it: `igm_run +experiment=params_past_v2` ➡️ results in `outputs/params_past_v2/` (**change the run folder at the end of the copied file**: `hydra: run: dir: outputs/params_past_v2`), with a copy of the full configuration (`.hydra/config.yaml`): every run is **reproducible**.
- All parameters and their defaults: https://igm-model.org (Modules)

---

# `oggm_shop` module to retrieve glacier data

Give an RGI v7 ID; IGM downloads (via [OGGM](https://oggm.org)) and maps to a 50 m / 100 m grid:

- surface topography (Copernicus DEM)
- ice thickness estimate (Millan et al. 2022)
- glacier outline (RGI v7)
- observed surface velocities (Millan et al. 2022)
- monthly climate 1901–present (GSWP3-W5E5)
- calibrated SMB parameters (to match Hugonnet et al. 2021 elevation change)

Everything ends up in `data/input_<RGI_ID>.nc`, cached for the next runs.

---

# Two ways to force the SMB

**1. `smb: method: simple`** — ELA + gradients, prescribed in time
- few, intuitive parameters; no climate needed
- good to explore sensitivity and to reconstruct the LIA

**2. `climate: method: oggm` + `smb: method: oggm`** — temperature-index model
- monthly temperature & precipitation → accumulation and melt
- observed climate extrapolated outside 1901–2019:

```yaml
clim_trend_array:
  - ["time", "delta_temp", "prec_scal"]
  - [ 2020,     1.3,     1.0]     # ≈ today (2000-2019 vs 1960-1990)
  - [ 2100,     4.0,     1.0]     # +4°C vs 1960-1990
```
- `melt_enhancer`: scales melt (uncertain melt factors)

---

# Ice-flow parameters

- **Ice softness** $A$ (`arrhenius`): softer ice → faster deformation
- **Basal sliding** (Weertman-type law): `tau_ref` = basal shear stress (MPa) needed to slide at 100 m/yr; lower `tau_ref` → more slippery bed → faster sliding

Tuned so that modelled speeds match the **observed** ones. Only the `physics` block of `iceflow` is yours:

```yaml
  iceflow:
    physics:
      sliding:
        tau_ref: 0.215        # explore 0.1-0.8
      viscosity:
        arrhenius: 78.0       # explore 30-150
    # --- emulator settings (numerics, unified): set for you, DO NOT MODIFY ---
```

---

<style scoped>table { font-size: 0.8em; }</style>

# Parameters to explore (and their ranges)

| Parameter | YAML key | Default | Explore | Calibrated on |
|---|---|---|---|---|
| Ice softness $A$ (higher = softer) | `iceflow` → `arrhenius` | 78 MPa⁻³ a⁻¹ | 30–150 | observed speeds |
| Basal sliding (higher = less sliding) | `iceflow` → `tau_ref` | 0.215 MPa | 0.1–0.3 | observed speeds |
| Melt (higher = more melt) | `smb` → `melt_enhancer` | 1.0 | 0.5–2 | today's extent |
| Pre-1901 climate | `climate` → `delta_temp` | –1 °C | –2 to 0 °C | LIA maximum |

One parameter at a time, one file per try!

---

# The ice-flow emulator `dahunet_edu`

- Solving the 3D (higher-order) ice-flow equations at each time step is **expensive**
- IGM uses instead a **convolutional neural network** :
 maps of `thk`, `usurf`, `arrhenius`, `tau_ref` ➡️ 3D ice velocity
- `dahunet_edu` was **trained off-line**, once and for all, on a large **catalogue of glaciers**, to reproduce the solutions of the physical equations
- We use it **as it is** (no re-training): take it as a trusted ice-flow model for mountain glaciers at ~50–250 m resolution
- Fast: 300 years of a mountain glacier in about a minute on CPU

⚠️ **do not modify** `numerics` and `unified` in the parameter files.

---

# Fast runs: coarse grid

**Grid coarsening**: `inputs.local.coarsening.ratio: 2` 

➡️ cells 2× larger (e.g. 100 m → 200 m), **4× fewer cells**, much faster runs
  - use `ratio: 2` to **test and calibrate** (many runs)
  - switch to `ratio: 1` to **finalize** (runs and figures of your report): copy your calibrated file (e.g. `params_past_final.yaml`) and set `ratio: 1` in it

 
---

# What you get

- `output.nc`: 2D distributed maps through time (thickness, speed, SMB, ...) 
- `output_ts.nc`: ice volume and area through time ➡️ time series
- `dashboard_XXXX.png`: quick-look images during the run

Helpers in `utils/helper.py` (used in the notebooks): `show_dashboard`, `compare_runs`, `animate_glacier_evolution`, `analyze_time_series`, `plot_flowline_evolution`, `compare_modeled_observed_velocity`

---

<!-- _class: invert center-img -->

# Visualizing results: `live_dashboard` module

![height:500px](fig/visu-dashboard.png)  

---

# Back in RenkuLab

Same as in Session 1:

1. renkulab.io → *My sessions* → **Resume** your paused session
2. Terminal: `cd ~/work/glamod && git pull` (latest course material)

New today: we run **IGM**, a command-line program. 

We work with **three tools**, side by side in JupyterLab.

---

# How to work with IGM: three tools

| What | Where | |
|---|---|---|
| **Set** the parameters | **YAML file** (`experiment/*.yaml`), opened in the JupyterLab text editor | one file = one experiment |
| **Run** IGM | **Terminal** (*File → New → Terminal*) | keeps running even if you close the browser tab |
| **Look** at the results | **Notebook**, reading `outputs/<run>/` | postprocessing only: never re-runs IGM |

Drag the tabs to split the screen.

---

<style scoped>table { font-size: 0.6em; } p, li { font-size: 0.8em; }</style>

# The terminal: the few commands you need

| Command | Does | Example |
|---|---|---|
| `pwd` | where am I? | `pwd` |
| `ls` | list files | `ls outputs/` |
| `cd` | change folder (`..` = up) | `cd ~/work/my-glacier` |
| `cp` | copy (`-r`: folder) | `cp params_past.yaml params_v2.yaml` |
| `mv` | rename / move | `mv old.yaml new.yaml` |
| `rm` | delete (`-r`: folder), **no undo!** | `rm -r outputs/params_v2` |

- **Tab** completes names, **↑** recalls commands, **Ctrl+C** stops a run
- `~` = home folder, `.` = current folder, `..` = parent folder
- Copy / rename / delete also work in the file browser (right-click)

---

# Run IGM in the terminal

```bash
cd ~/work/my-glacier          # the folder that contains experiment/
igm_run +experiment=params_past
```

- `+experiment=params_past` → reads `experiment/params_past.yaml`
- Results go to `outputs/params_past/`: **the run folder is set at the end of each file** (`hydra: run: dir:`): when you copy a file, change it, or the results overwrite the previous run
- Stop a run: **Ctrl+C**. Recall the last command: **↑**

Each run folder contains `output.nc` (maps), `output_ts.nc` (volume, area), `dashboard_*.png` (quick looks) and `.hydra/` (**all** parameters used).

---

# Look at the results in a notebook

```python
from helper import *                      # course plotting helpers (utils/helper.py)
show_dashboard("outputs/params_past")     # last dashboard images
compare_runs(["outputs/params_past", "outputs/params_past_v2"],
             ["v1", "v2"])                 # ice volume through time
```

- Re-run the notebook as often as you like: it only **reads** the run folders
- Figures for the report: `plt.savefig("figures/fig_volume.png", dpi=200, bbox_inches="tight")`

**You may back up your work** at the end of each afternoon (Session 1, Step 4): `cd ~/work && tar czf my-glacier.tar.gz --exclude=outputs --exclude=data my-glacier` → *Download* → UNIL OneDrive. Run folders can always be re-created.

---

# When something goes wrong

| Symptom | Fix |
|---|---|
| `igm_run: command not found` | Wrong launcher/environment → use the `glamod` launcher |
| `Parameter 'experiment' does not exist` | You are not in the folder that contains `experiment/` → `cd` there |
| `FileNotFoundError: outputs/...` in the notebook | The run is not finished, or the folder name differs from the parameter file name |
| Kernel dies / very slow | Out of memory: restart the kernel, shorten the run, close other notebooks |
| Data download (`oggm_shop`) fails | Retry: the server may be busy when 20 students download at once |

Ask the assistant!

---

# The mini-project: model your own glacier

> **How did my glacier evolve since the Little Ice Age (LIA, ~1850), and how will it evolve until 2100 under different climate scenarios?**

| Part | What | When |
|---|---|---|
| **A. Common base** (everyone) | Present → ice-flow calibration → Past (SMB calibration) → Future | **today** |
| **B. Explorations** | simple SMB model, response time | **today** |
| | moraines (particles), figures for the report | Session 3 |

Then: the report (Lecture 3). Deadline: *to be announced*

---

# Hands-on: follow `tutorial.ipynb` ⚠️

**Your guide for the afternoon is the notebook `tutorial.ipynb`**: follow it from top to bottom, it tells you which file to edit and which command to run.

| Step | (= section of `tutorial.ipynb`) |
|---|---|
| 0 | **Your own copy** of the template folder, your glacier |
| 1 | **Present**: download the data, check the initial state |
| 2 | **Calibration 1**: ice-flow parameters (`arrhenius`, `tau_ref`) on the observed speeds |
| 3 | **Past**: LIA → today, **Calibration 2**: SMB (`melt_enhancer`) |
| 4 | **Future**: climate scenarios 2020–2100 |
| 5–6 | A simple SMB model (ELA), response time |

All runs in the **terminal**; the notebook **only plots**.

---

# Step 0 — Your own copy of the template

Once, in a terminal (**your** folder, outside the course material, which may be updated):

```bash
cp -r ~/work/glamod/session2/my-glacier ~/work/my-glacier
cd ~/work/my-glacier
```

1. Open **`tutorial.ipynb`**, run the first cell; drag the terminal next to it
2. Go through it with the **Rhone Glacier** (default in all files), then do the same with **your** glacier: put its RGI ID (`RGI_ID`) in **each** file of `experiment/`

---

# The template: `my-glacier/`

```
my-glacier/                         ← run igm_run from HERE
├── tutorial.ipynb                  YOUR GUIDE: follow it step by step (plots only)
├── experiment/
│   │                               (each file: YOUR RGI ID + GRID + ICE FLOW + its experiment)
│   ├── params_present.yaml         Step 1: 1-year check run
│   ├── params_sensitivity.yaml     Step 2: grid of ice-flow parameters (igm_run -m)
│   ├── params_past.yaml            Step 3: 1700-2020, climate-driven SMB
│   ├── params_future.yaml          Step 4: 2020-2100, climate scenarios (igm_run -m)
│   ├── params_past_simple.yaml     Step 5: 1700-2020, ELA-based SMB
│   ├── params_response_time.yaml   Step 6: oscillating ELA (your own module in user/)
│   └── params_past_moraines.yaml   Session 3: past run + debris particles
└── outputs/, figures/, data/       created by IGM and the notebook
```

---

# Step 1 — Present: check the initial state

```bash
cd ~/work/my-glacier
igm_run +experiment=params_present
```

- Downloads the data (once) to `data/`, gives today's modelled speeds
- Notebook: `plot_input_data`, `plot_climate` (area, volume, thickness, observed speeds, climate)
- Compare with what you know: GLAMOS, swisstopo thickness map

---

# Step 2 — Calibration 1: the ice flow

`compare_modeled_observed_velocity`: too fast / too slow with the default parameters? Then all combinations `arrhenius` × `tau_ref` (`-m` = multirun, one 1-year run per combination, ≈ 3 min):

```bash
igm_run -m +experiment=params_sensitivity \
   processes.iceflow.physics.viscosity.arrhenius=25,50,100 \
   processes.iceflow.physics.sliding.tau_ref=0.1,0.2,0.4,0.8
```

- ➡️ `outputs/params_sensitivity/0` … `11`; `compare_velocity_misfit`: maps of the speed difference, look for **bias ≈ 0**, small RMSE
- Several combinations equally good? (trade-off deformation / sliding)
- ✅ Put your values in the `iceflow` block of **all** files (Rhone: `arrhenius` 30, `tau_ref` 0.315)

---

# Step 3 — Past: from the LIA to today (Calibration 2)

Trick: **start in 1700 from today's geometry**, cool the climate ➡️ LIA maximum (~1850), then retreat with the observed climate 1901–2019.

```bash
igm_run +experiment=params_past
```

- Keep the pre-1901 climate (`clim_trend_array`) of `params_past.yaml`
- **Calibrate `melt_enhancer`** (start: 0.8, explore 0.5–2), one file per try (`params_past_v2.yaml` ➡️ `outputs/params_past_v2/`, set at the end of the file)
- Validate: LIA moraines, Siegfried map, today's ice volume
- ✅ Put your `melt_enhancer` in `params_future.yaml` (and `params_past_moraines.yaml`)

---

# Step 4 — Future: 2020–2100

`delta_temp` is relative to **1960–1990** (`plot_climate` prints the offsets):
- **2020** (in the file): today's offset (≈ +1.3 °C for the Rhone Glacier)
- **2100** (command line): 1981–2010 offset **+** scenario warming (CH2018):

| RCP2.6 / SSP1-2.6 | RCP4.5 / SSP2-4.5 | RCP8.5 / SSP5-8.5 |
|---|---|---|
| ≈ +0.6 to +1.9 °C | ≈ +2.0 to +3.3 °C | ≈ +3.3 to +5.4 °C |

```bash
igm_run -m +experiment=params_future processes.climate.oggm.clim_trend_array.2.1=1.3,2.0,3.4,5.1
```

➡️ `outputs/params_future/0` … `3`. When does the glacier (nearly) disappear?

---

# Step 5 — A simple SMB model: ELA-based

```bash
igm_run +experiment=params_past_simple
```

- Prescribe the **ELA history** yourself (`params_past_simple.yaml`): today's ELA (snowline on map.geo.admin.ch, GLAMOS); LIA ELAs ≈ 100–200 m lower
- **Try to reproduce the volume curve** of your calibrated past run (`compare_runs`) by playing with the ELAs
- Which one is easier to calibrate? Which one would you trust for the future, and why?

Step 6 — Response time: see the end of `tutorial.ipynb` (and Lecture 3)
