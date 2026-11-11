---
marp: true
theme: default
class: invert
backgroundColor: black
color: white
math: katex
---

# Session 3 — Lecture 3: Experiment deepening and reporting

Guillaume Jouvet

---

# Course format: 3 afternoons × 4 hours + 1 extra

| | Hour 1 | Hour 2 | Hour 3 | Hour 4 |
|---|---|---|---|---|
| **Week 1** | Glacier processes | Glacier modelling | *Tutorial* | *Glacier selection* |
| **Week 2** | Intro to IGM, *basic workflow* (Rhone, then your glacier) | *Project* | *Project* | *Project* |
| **Week 3** | **Experiment deepening, report** | **Project** | **Project** | **Project** |
| *(Week 4* | *dates and rate add-on* | *G. King's course* | *Project* | *Project)* |
 


---

# Part 1 — Goals of your mini-project

**Main question**: how did *your* glacier evolve since the Little Ice Age, and how will it evolve until 2100?

1. **Present**: is the model state realistic today? (speeds vs. observations)
2. **Past**: can the model reproduce the **LIA maximum (~1850)** and the retreat to today? (moraines, old maps, GLAMOS length changes)
3. **Future**: 2020–2100 under climate scenarios (RCP 2.6, 4.5, 8.5).

➡️ **Session 2, basic workflow**: a calibrated model of your glacier & future
➡️ **Session 3, experiment deepening**: LIA extent, response time, moraines, climate controls, committed loss, lake appear., debris covering, societal impacts.

---

# Finish the calibration: tips

| Control ↓ &nbsp; Target → | Speeds | Volume today | 1850 extent |
|---|:-:|:-:|:-:|
| **Ice flow**: `arrhenius`, `tau_ref` | ●●● | ● | ●● |
| **Melt**: `melt_enhancer` | | ●●● | ●● |
| **Past climate**: pre-1901 `delta_temp` | | ● | ●●● |

- No single parameter fits all targets: e.g. a `melt_enhancer` that fits today's volume may give a wrong 1850 extent ➡️ **iterate** between the three controls to find the **best compromise**
- One parameter at a time; no perfect fit: **discuss** the mismatch in the report
- Tests with `ratio: 2`, **final runs with `ratio: 1`**

---

 
# Part 2 — Experiment deepening

1. **First, fine-tune the model with the LIA extent**: does the model reach the mapped 1850 extent? ➡️ closer calibration of the past, then re-run the future
2. **Then, work on optional extensions**, by increasing difficulty: **A–E** (questions with the runs you know), then **F) response time** and **G) moraines**

Follow **`tutorial.ipynb`, sections 6–8** (Session 3 part). Keep working in **your** `~/work/my-glacier`; update the course material and the notebook (details: last slide):

```bash
cd ~/work/glamod && git restore . && git pull
cd ~/work/my-glacier && mv tutorial.ipynb tutorial-session2.ipynb
cp ~/work/glamod/session2/my-glacier/tutorial.ipynb .
```

---

# The Little Ice Age (LIA) in the Alps

![width:1000px](fig/lia-temperature.png)

- ~1300–1860: cooler summers (~0.5–1 °C) ➡️ last maximum **~1850**
- Since then: retreat, accelerating since the 1980s

---

# After the LIA: the Rhone Glacier from Gletsch

![height:520px](fig/rhone-gletsch-1856-2008.png)

---

# LIA extent: the data

The LIA maximum (~1850) is the best-documented past state: **moraines**, first topographic maps

- **Swiss Glacier Inventory 1850** (SGI 1850, [GLAMOS](https://www.glamos.ch/en/downloads); Maisch et al., 2000): outlines of all Swiss glaciers in 1850 ➡️ downloaded by the notebook
- **Historical maps**, [map.geo.admin.ch](https://map.geo.admin.ch) *Journey through time*: Dufour (1845–1865), Siegfried (1870–1926)
- **Moraines**: swissALTI3D relief, SWISSIMAGE: youngest, fresh, unvegetated ridges in front of the glacier
- **Front retreat** since ~1880: GLAMOS length changes (factsheet of your glacier)

---

# LIA extent: calibrate the past

![bg right:42% contain](fig/lia-extent-rhone.png)
  
- Wrong **in 1850** 
➡️ tune pre-1901 `delta_temp`
- Wrong **today** 
➡️ tune `melt_enhancer`
- Front too thin 
➡️ tune ice flow
- No single parameter fits all ➡️ iterate, best compromise
- Then redo the future runs

---

# Extensions A–E 

- **A) Committed loss**: if today's climate were kept unchanged, when would the glacier stabilise (beyond 2100), and at what volume compared with today?
- **B) Climate controls**: what are the main controls of the future evolution? Can a +1 °C warming be compensated by more precipitation? If so, by how much?
- **C) Society**: what could be the direct societal impacts of the changes in the catchment hydrology caused by the future glacier retreat, and how do they differ between scenarios?
- **D) Disconnection and lakes**: is your glacier expected to split into several branches as it retreats? Are proglacial lakes expected to form in bed overdeepenings (`topg`)?
- **E) Debris cover**: if your glacier is debris-covered, melt is probably reduced: can it survive longer than a clean glacier? Can this hypothesis be tested?

➡️ More demanding: **F) response time** (your own IGM module), **G) moraines** (particles)

---

# F) Response time

The response time is tested with an IGM **user module** for the SMB (instead of the built-in SMB used so far), `smb_oscillating.py`: the ELA oscillates, the glacier follows... with a delay

```python
def update(cfg, state):                        # called at each time step
    p = cfg.processes.smb_oscillating
    t = float(state.t.numpy()) - cfg.processes.time.start
    state.ela = state.ela_mean - p.amplitude * math.sin(2 * math.pi * t / p.period)
    state.smb = state.usurf - state.ela
```

```bash
igm_run +experiment=params_response_time
```

---

# F) Response time: example (Rhone Glacier)

![height:450px](fig/response-time-rhone.png)

---

# F) Glacier response times

- **Lag** between the ELA minimum and the volume peak (`plot_response_time`)
- **Other periods** $P$ (50–800 yr): how do the lag and the amplitude change? Towards which value does the lag tend for long periods? 
➡️ **response time** $\tau$. Compare with $\tau \approx H/(-b_{terminus})$
- **Size**: do you expect $\tau$ to grow with the volume? 
- **Conclude**: what happens to short climate variations? to long-term trends?

---

# G) Moraines: tracking debris with particles

![width:1000px](fig/particles-rhone.png)

Particles carried by the ice flow (Rhone Glacier, 1510 → 2000) 

➡️ Passive "particles" can be added on top of the simulation to model the supply and transport of supraglacial debris (IGM module `particles`).

---

# G) Moraines: reproduce the LIA moraines

![bg right:38% contain](fig/moraines-rhone.png)

Retreat after 1850: debris left on ice-free terrain ➡️ **modelled LIA moraines**

- Run `params_past_moraines`, then `plot_moraines`
- Compare with **aerial photos / relief** (map.geo.admin.ch)
- **Controls**: `seeding`, calibration of the past
- Limits: debris comes from rock walls; moraine building not modelled

---


# Tips for a good experiment design

- **One question → one (or a few) experiments → one figure**
- Change **one thing at a time** (ELA, `delta_temp`, `melt_enhancer`, `tau_ref`, ...)
- Keep **one YAML file per experiment**, with a meaningful name (`params_past_ref.yaml`, `params_ssp585.yaml`, ...)
- Write down for each run: what you changed, why, and the outcome (a simple table/log)
- Explore **uncertainty**: run a small ensemble (e.g. 3 values of a parameter) rather than one "perfect" run
 
---

# Part 3 — The report

- **≤ 1500 words** (excluding captions and references), PDF
- Written like a short scientific paper, for instance with this structure (a suggestion, not a strict rule):

1. **Introduction**: why model glaciers? state of knowledge, your research question(s)
2. **Study site**: location, characteristics, history, available observations
3. **Methods**: the model (what it does, *not* how you ran it: no code, no YAML), data, experiments, **this is not a technical documentation**
4. **Results**: figures, what you found
5. **Discussion**: answer your questions, limitations, uncertainties
6. **Conclusion** + references

---

# Figures convey your results

- Typical figures: map of the glacier (thickness/speed) at key dates, **time series** of ice volume, **flowline profiles**, model vs. observations
- Every figure: axes with units, legend, readable font, a caption
- Compare model with **observations** whenever possible (LIA extent, GLAMOS length changes, observed speeds)
- Fewer, better figures (4–6 is a good number): make a selection, do not show results you do not discuss. Quality over quantity
- Customise your figures by adapting the provided plotting functions (`utils/helper.py`)
- Get inspired by the figures of published papers, e.g. https://doi.org/10.3189/2013AoG63A434

---

# Discussion: be critical

- Which results are robust? Which depend strongly on a poorly known parameter?
- Main sources of uncertainty: initial ice thickness, SMB/climate forcing, sliding, starting from today's geometry for the past, 100 m resolution, ...
- Does the model reproduce the observations? If not, why?
- What would you do with more time/data?
 
---

# Now: work on your project

Today:
- Finish the basic workflow on your glacier (Session 2)
- Fine-tune the calibration of the past with the 1850 extent, then re-run the future scenarios
- Then: extensions (A–G) suitable for your glacier
- Draft your figures
- Start your report (outline, study site, methods)

Work in `~/work/my-glacier/`.

---

# Updating the course material (glamod)

```bash
cd ~/work/glamod
git restore .     # discard any change made in glamod (it is not your project!)
git pull          # get the latest version of the course
```

- `~/work/glamod`: the **course material**, updated by `git pull` (incl. `utils/helper.py`)
- `~/work/my-glacier`: **your project**, a **local copy** made last week: `git pull` does **not** update it

---

# Updating your project (my-glacier)

⚠️ The new template files (`tutorial.ipynb`, `experiment/*.yaml`) are in `~/work/glamod/session2/my-glacier/`: copy only what you need, **without overwriting your own work** (calibrated values, RGI ID, notes)

```bash
cd ~/work/my-glacier
mv tutorial.ipynb tutorial-session2.ipynb        # keep your Session 2 notebook
cp ~/work/glamod/session2/my-glacier/tutorial.ipynb .
```

For a YAML file, compare first, then copy the new lines by hand:

```bash
diff experiment/params_past_moraines.yaml \
     ~/work/glamod/session2/my-glacier/experiment/params_past_moraines.yaml
```
