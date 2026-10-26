---
marp: true
theme: default
class: invert
backgroundColor: black
color: white
---

# Glacier Modelling

## Session 3 — Lecture 3: Project goals and reporting

Guillaume Jouvet

---

# Course format: 3 afternoons × 4 hours + 1 extra

| | Hour 1 | Hour 2 | Hour 3 | Hour 4 |
|---|---|---|---|---|
| **Week 1** | Glacier processes | Glacier modelling | *Tuto 1* | *Glacier selection* |
| **Week 2** | Intro to IGM, *Tuto 2* | *Project* | *Project* | *Project* |
| **Week 3** | **Project goals, report** | **Project** | **Project** | **Project** |
| *(Week 4* | *dates and rate add-on* | *G. King's course* | *Project* | *Project)* |
 


---

# Part 1 — Goals of your mini-project

**Main question**: how did *your* glacier evolve since the Little Ice Age, and how will it evolve until 2100?

1. **Present**: is the model state realistic today? (thickness, speed vs. observations)
2. **Past**: can the model reproduce the **LIA maximum (~1850)** and the retreat to today? → calibration / validation with moraines, Siegfried map, GLAMOS length changes
3. **Future**: glacier evolution 2020–2100 under ≥ 2 climate scenarios (e.g. SSP1-2.6 ≈ +1.5°C, SSP2-4.5 ≈ +2.5°C, SSP5-8.5 ≈ +4.5°C)

Optional: your own question (moraines with particles, sensitivity to sliding/softness, response time, lake formation, ...).

---

# A good experiment design

- **One question → one (or a few) experiments → one figure**
- Change **one thing at a time** (ELA, `delta_temp`, `melt_enhancer`, `tau_ref`, ...)
- Keep **one YAML file per experiment**, with a meaningful name (`params_past_ref.yaml`, `params_ssp585.yaml`, ...)
- Write down for each run: what you changed, why, and the outcome (a simple table/log)
- Explore **uncertainty**: run a small ensemble (e.g. 3 values of a parameter) rather than one "perfect" run

---

# Part 2 — The report

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

- Typical figures: map of the glacier (thickness/speed) at key dates, **time series** of volume/area/length, **flowline profiles**, model vs. observations
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
- [ ] Past run that reaches a plausible LIA maximum and today's extent
- [ ] ≥ 2 future scenarios
- [ ] Figures drafted
- [ ] Report outline written

`project/project.md` — `project/my-glacier/` (YAML + terminal + `analysis.ipynb`)
