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

➡️ **Session 2, basic workflow**: a calibrated model of your glacier, used for the future
➡️ **Session 3, experiment deepening**: LIA extent, then response time, moraines

---

# Finish the calibration: tips

| Problem | Try |
|---|---|
| Speeds too high / too low | refine the grid of `params_sensitivity` (`igm_run -m`) around your best values |
| No LIA advance / too large | colder / warmer pre-1901 `delta_temp` (`clim_trend_array`) |
| Glacier grows without limit / disappears | more / less melt: `melt_enhancer` |

- **One parameter at a time**, keep your calibrated values; do not expect perfect fit, **discuss** the mismatch in the report anyway
- Tests with `ratio: 2`, **final runs with `ratio: 1`**

---

 
# Part 2 — Experiment deepening

1. **First, everyone — fine-tune the model with the LIA extent**: does the model reach the mapped 1850 extent? ➡️ closer calibration of the past, then re-run the future
2. **Then, if time allows — extensions**: **response time**, **moraines**, or one of the other questions (optional extensions, later)

Keep working in **your** `~/work/my-glacier` (from Session 2). Just update the course material and add the Session 3 notebook to it:

```bash
cd ~/work/glamod && git pull
cp ~/work/glamod/session3/tutorial-session3.ipynb ~/work/my-glacier/
```

---

# The Little Ice Age (LIA) in the Alps

![width:1100px](fig/lia-temperature.png)

- ~1300–1860: cooler summers (~0.5–1 °C) ➡️ repeated glacier advances, last maximum **~1850**
- Since then: retreat, accelerating since the 1980s; the LIA moraines mark the 1850 extent

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

`compare_lia_extent`: model in 1850 vs SGI 1850 (red), today (black)

Rhone, `melt_enhancer` 0.8: the front goes **too far**

- Wrong **in 1850** ➡️ pre-1901 `delta_temp`
- Wrong **today** ➡️ `melt_enhancer`
- Front too thin ➡️ ice flow
- Then redo the future runs

---

# Response time

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

# Response time: example (Rhone Glacier)

![height:250px](fig/response-time-rhone.png)

- The volume peaks **~45 years** after the ELA minimum (period 200 yr, amplitude 200 m): close to $P/4$, the period is too short
- Longer periods: $\tau \approx$ 75 yr ($P$ = 400 yr), 55 yr ($P$ = 800 yr); theory $H/(-b_{terminus}) \approx$ 380 m / 6 m/yr $\approx$ 65 yr

---

# Response time: what to do

1. **Find the response time** from the lag: $\tau = \frac{P}{2\pi}\tan\left(\frac{2\pi\,\text{lag}}{P}\right)$ (first-order glacier, printed by `plot_response_time`)
2. **Test other periods** $P$ (50 … 800 yr, run 2 periods): lag and amplitude? Compare $\tau$ with the theory $\tau \approx H/(-b_{terminus})$ (Lecture 1a)
3. **Relate $\tau$ to the size / volume** of your glacier: compare with other students' glaciers
4. **Conclude**: a glacier is a **low-pass filter**: it damps climate variations shorter than $\tau$, integrates the long-term trends ➡️ committed loss

---

# Moraines: tracking debris with particles

![width:1000px](fig/particles-rhone.png)

Particles carried by the ice flow (Rhone Glacier, 1510 → 2000) 

➡️ Passive "particles" can be added on top of the simulation to model the supply and transport of supraglacial debris (IGM module `particles`).

---

# Moraines: reproduce the LIA moraines

![bg right:38% contain](fig/moraines-rhone.png)

Retreat after 1850: debris left on ice-free terrain ➡️ **modelled LIA moraines**

- Run `params_past_moraines`, then `plot_moraines`
- Compare with **aerial photos / relief** (map.geo.admin.ch)
- **Controls**: `seeding`, calibration of the past
- Limits: debris comes from rock walls; moraine building not modelled

---

# Optional extensions

- **Climate controls**: what are the main controls of the future evolution? Can a warming of +1 °C be compensated by more precipitation? If so, by how much?
- **Committed loss**: if today's climate were kept unchanged, when would the glacier stabilise (beyond 2100), and at what volume compared with today?
- **Disconnection and lakes**: is your glacier expected to split into several branches as it retreats? Are proglacial lakes expected to form in bed overdeepenings (`topg`)?
- **Debris cover**: if your glacier is debris-covered, melt is probably reduced: can it therefore survive longer than a clean glacier? Can this hypothesis be tested?
- **Society**: what could be the direct societal impacts of the changes in the catchment hydrology caused by the future glacier retreat, and how do they differ between scenarios?

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
- If time allows: response time, moraines, or other extensions
- Draft your figures
- Start your report (outline, study site, methods)

Work in `~/work/my-glacier/`. 