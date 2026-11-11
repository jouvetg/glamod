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

<style scoped>section { font-size: 1.6em; }</style>

# Part 1 — Goals of your mini-project

**Main question**: how did *your* glacier evolve since the Little Ice Age, and how will it evolve until 2100?

1. **Present**: is the model state realistic today? ( speed vs. observations)
2. **Past**: can the model reproduce the **LIA maximum (~1850)** and the retreat to today? → calibration / validation with moraines, old maps, recorded GLAMOS length changes
3. **Future**: glacier evolution 2020–2100 under climate scenarios (e.g. RCP 2.6 ≈ +1.5°C, RCP 4.5 ≈ +2.5°C, RCP 8.5 ≈ +4.5°C), or no warming (committed loss).

➡️ **Session 2, basic workflow** : one model of your glacier, calibrated on the speeds and on the past, used for the future
➡️ **Session 3, experiment deepening**

---

# Finish the calibration: tips

| Problem | Try |
|---|---|
| Speeds too high / too low | refine the grid of `params_sensitivity` (`igm_run -m`) around your best values |
| No LIA advance / too large | colder / warmer pre-1901 `delta_temp` (`clim_trend_array`) |
| Glacier grows without limit / disappears | more / less melt: `melt_enhancer` |

- **One parameter at a time**, keep your calibrated values in the ✅ boxes of `tutorial.ipynb`; no perfect fit: **discuss** the mismatch
- Tests with `ratio: 2`, **final runs with `ratio: 1`**

---

 
# Part 2 — Experiment deepening

**Suggestions** to deepen your study:

| | Question |
|---|---|
| **D1** | **LIA extent**: does the model reach the mapped 1850 extent? ➡️ closer calibration of the past |
| **D2** | **Response time**: how fast does your glacier react? |
| **D3** | **Moraines**: where does the debris go? (particles; if relevant) |

Keep working in **your** `~/work/my-glacier` (from Session 2). Just update the course material and add the Session 3 notebook to it:

```bash
cd ~/work/glamod && git pull
cp ~/work/glamod/session3/tutorial-session3.ipynb ~/work/my-glacier/
```

---

# D1 — LIA extent: the data

The LIA maximum (~1850) is the best-documented past state: **moraines**, first topographic maps

- **Swiss Glacier Inventory 1850** (SGI 1850, [GLAMOS](https://www.glamos.ch/en/downloads); Maisch et al., 2000): outlines of all Swiss glaciers in 1850 ➡️ downloaded by the notebook
- **Historical maps**, [map.geo.admin.ch](https://map.geo.admin.ch) *Journey through time*: Dufour (1845–1865), Siegfried (1870–1926)
- **Moraines**: swissALTI3D relief, SWISSIMAGE: youngest, fresh, unvegetated ridges in front of the glacier
- **Front retreat** since ~1880: GLAMOS length changes (factsheet of your glacier)

---

# D1 — LIA extent: calibrate the past

![bg right:42% contain](fig/lia-extent-rhone.png)

`compare_lia_extent`: model in 1850 vs SGI 1850 (red), today (black)

Rhone, `melt_enhancer` 0.8: the front goes **too far**

- Wrong **in 1850** ➡️ pre-1901 `delta_temp`
- Wrong **today** ➡️ `melt_enhancer`
- Front too thin ➡️ ice flow
- Then redo the future runs

---

# D2 — Response time: your own IGM module

`user/code/processes/smb_oscillating.py`: the ELA oscillates, the glacier follows... with a delay

```python
def update(cfg, state):                        # called at each time step
    p = cfg.processes.smb_oscillating
    t = float(state.t.numpy()) - cfg.processes.time.start
    state.ela = state.ela_mean - p.amplitude * math.sin(2 * math.pi * t / p.period)
    smb = state.usurf - state.ela              # SMB linear in elevation
    state.smb = ...
```

```bash
igm_run +experiment=params_response_time
```

---

# D2 — Response time: example (Rhone Glacier)

![height:250px](fig/response-time-rhone.png)

- The volume peaks **~45 years** after the ELA minimum (period 200 yr, amplitude 200 m)
- Other periods (50, 100, 400 yr): copy the file (`params_response_100.yaml`), set `period: 100` in it: how do lag and amplitude change? Compare with $\tau \approx H/(-b_{terminus})$ (Lecture 1a) and the tutorial
- Read the module: how to extend IGM with `initialize`, `update`, `finalize`

---

# D3 — Moraines: tracking debris with particles

![width:700px](fig/particles-rhone.png)

Particles carried by the ice flow (Rhone Glacier, 1510 → 2000) ➡️ **moraines**, **debris cover**

`params_past_moraines.yaml` = your **calibrated** past run + `particles` (seeded every 10 yr in the accumulation area)

---

# D3 — Moraines: questions

```bash
igm_run +experiment=params_past_moraines
```

Notebook: `plot_particles(run, year=...)`, and `rh_min=0.95` for the debris at the surface only

- Where do the particles re-emerge at the surface? Do they form medial / lateral moraines?
- Compare with the moraines on aerial photos (map.geo.admin.ch) and with the LIA moraines
- Particles left on ice-free terrain: debris deposited where the glacier retreated
- How old is the debris when it reaches the front?

---

# C. Extensions (optional, discuss them with us first)

- **Uncertainty**: small ensembles (melt factor, initial thickness, `prec_scal`, CH2018 ranges)
- **Precipitation**: ±20 % precipitation vs ±1 °C?
- **Committed loss**: today's climate kept, when does the glacier stabilize (beyond 2100)?
- **Tributaries and lakes**: disconnections, proglacial lakes in bed overdeepenings (`topg`)
- **Validation**: GLAMOS mass balance series, swisstopo thickness
- **Your own question**: hazards, water resources, landscape evolution...


---

# A good experiment design

- **One question → one (or a few) experiments → one figure**
- Change **one thing at a time** (ELA, `delta_temp`, `melt_enhancer`, `tau_ref`, ...)
- Keep **one YAML file per experiment**, with a meaningful name (`params_past_ref.yaml`, `params_ssp585.yaml`, ...)
- Write down for each run: what you changed, why, and the outcome (a simple table/log)
- Explore **uncertainty**: run a small ensemble (e.g. 3 values of a parameter) rather than one "perfect" run


 
---

# Part 3 — The report

- **≤ 1500 words** (excluding captions and references), PDF
- Written like a short scientific paper:

1. **Introduction**: why model glaciers? state of knowledge, your research question(s)
2. **Study site**: location, characteristics, history, available observations
3. **Methods**: the model (what it does, *not* how you ran it: no code, no YAML), data, experiments
4. **Results**: figures!
5. **Discussion**: answer your questions, limitations, uncertainties
6. **Conclusion** + references

---

# Methods: describe the model, not the code

❌ "We ran `igm_run +experiment=params_past` with `smb: method: oggm` and `melt_enhancer: 1.4`."

✅ "Ice flow is modelled with the Instructed Glacier Model (IGM; Jouvet & Cordonnier, 2023), which solves a higher-order approximation of the Stokes equations with a physics-informed neural network. The surface mass balance is computed with a monthly temperature-index model (Maussion et al., 2019), forced by the GSWP3-W5E5 climate; melt was scaled by a factor of 1.4 to reproduce the observed front position in 2020."

A reader who does not know IGM must understand what you did, and be able to reproduce it.

---

# Figures convey your results

- Typical figures: map of the glacier (thickness/speed) at key dates, **time series** of ice volume, **flowline profiles**, model vs. observations
- Every figure: axes with units, legend, readable font, a caption that says what to see
- Compare model with **observations** whenever possible (LIA extent, GLAMOS length changes, observed speeds)
- Fewer, better figures (4–6 is a good number)

Inspiration: Jouvet et al. (2011) https://doi.org/10.3189/002214311798843359 — https://doi.org/10.3189/2013AoG63A434

---

# Discussion: be critical

- Which results are robust? Which depend strongly on a poorly known parameter?
- Main sources of uncertainty: initial ice thickness, SMB/climate forcing, sliding, starting from today's geometry for the past, 100 m resolution, ...
- Does the model reproduce the observations? If not, why?
- What would you do with more time/data?

---

# Evaluation

| Criterion | Weight |
|---|---|
| A. Research question, context, literature | 25% |
| B. Modelling tools and numerical experiments | 25% |
| C. Results and discussion | 25% |
| D. Form and quality (structure, figures, writing) | 10% |
| Handed in on time | |

Deadline: **to be announced**. Questions? Ask the assistant during the project hours.

<!-- TEACHER NOTE: weights taken from 2025 AGRADING.md (sum = 85%); confirm the remaining 15% (on-time / oral?) and the deadline. -->

---

# Now: work on your project

Checklist for today:
- [ ] Basic workflow done on your glacier (Session 2)
- [ ] ≥ 1 deepening (D1 LIA extent, D2 response time, D3 moraines): not all three
- [ ] Figures drafted
- [ ] Report outline written

`~/work/my-glacier/` (your copy: YAML + terminal + `tutorial.ipynb`, `tutorial-session3.ipynb`)

**Back up** at the end of each afternoon (`cd ~/work && tar czf my-glacier.tar.gz --exclude=outputs --exclude=data my-glacier` → *Download* → OneDrive): RenkuLab sessions are not permanent storage. Data problem? Delete `data/input_<RGI_ID>.nc` and run again.
