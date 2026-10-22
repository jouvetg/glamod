---
marp: true
theme: default
class: invert
backgroundColor: black
color: white
---

# Glacier Modelling

## Session 1 — Lecture 1a: Glacier processes

Guillaume Jouvet

![height:200px](fig/rhone-mod-lq.png)

---

# Today (Session 1)

| | |
|---|---|
| Hour 1 | Lecture 1a: Glacier processes |
| Hour 2 | Lecture 1b: Glacier modelling — and a cold case solved with a glacier model |
| Hour 3 | Set up RenkuLab (≈ 15 min), then Tutorial 1: a simple glacier model (`session1/1c-tutorial-1-sia-model.ipynb`) |
| Hour 4 | Choose your glacier for the mini-project (`session1/1d-choose-your-glacier.md`) |

---

# Course objectives

- Provide the basics of glacier processes
- Introduce numerical methods to model glacier evolution
- Perform practical numerical experiments with IGM, on **your own glacier**

---

# Course format: 3 afternoons × 4 hours

| | Hour 1 | Hour 2 | Hour 3 | Hour 4 |
|---|---|---|---|---|
| **Session 1** | Glacier processes | Glacier modelling + an application (the Ebener case) | RenkuLab setup + Tutorial 1 (simple model) | Choose your glacier |
| **Session 2** | Introduction to IGM + first IGM runs (Tutorial 2) | Mini-project | Mini-project | Mini-project |
| **Session 3** | Project goals, how to write the report | Mini-project | Mini-project | Mini-project |

All computations run **online** (RenkuLab, set up today in hour 3), nothing to install on your laptop.

---

# Resources

- If there is one resource to remember, it is the website https://www.antarcticglaciers.org, which covers glacier processes in depth.

- To dive into the mechanics and the maths, I recommend the "Karthaus Summer School Lecture Notes":
https://link.springer.com/book/10.1007/978-3-030-42584-5

- IGM documentation: https://igm-model.org

---

# Main processes governing glacier evolution

![width:60%](fig/scheme_glacier.png)

---

# Glacial processes governing glacier evolution

- Climatic surface mass balance (= accumulation – ablation)
- Ice thermo-dynamics
- Mass conservation
- Subglacial hydrology
- Debris / sediment transport

---

# Mass balance processes

![height:450px](https://www.antarcticglaciers.org/wp-content/uploads/2018/11/glaciers-as-a-system.png)

Picture from www.antarcticglaciers.org

---

# Surface mass balance (SMB) and the ELA

- **Accumulation**: snowfall (+ avalanches, wind drift, refreezing)
- **Ablation**: melt (mostly), sublimation, calving
- **SMB** = accumulation − ablation, in m of ice (or water) equivalent per year
- **Equilibrium Line Altitude (ELA)**: elevation where SMB = 0
  - above: accumulation area (SMB > 0)
  - below: ablation area (SMB < 0)
- In the Alps today, ELA ≈ 3000–3300 m; during the Little Ice Age (~1850), ≈ 100–200 m lower

More: https://www.antarcticglaciers.org/glacier-processes/mass-balance/introduction-glacier-mass-balance/


---

# SMB primarily depends on altitude

![height:500px](https://npshistory.com/publications/geology/pp/715-A/images/fig1.jpg)


---

# Modelling SMB: two families

1. **Elevation-based** (simplest): SMB depends linearly on elevation, with an ELA and two gradients (ablation / accumulation)
$$ SMB(z) = \beta\,(z - z_{ELA}), \qquad \text{capped at } b_{max}$$

2. **Climate-based** (temperature-index / positive degree-day): melt ∝ positive temperatures, accumulation = solid precipitation
$$ SMB = \underbrace{P_{solid}}_{\text{accumulation}} - \underbrace{f_{melt}\sum \max(T - T_{melt},0)}_{\text{melt}}$$

We will use both in IGM.

---

# Observing glacial motion

Timelapse cameras have captured glacial motion:

- https://www.youtube.com/watch?v=TWGR6FxFlt8

- https://www.youtube.com/watch?v=he5QzhE7_g4&t=55s

- https://www.youtube.com/watch?v=iaULB6O-JNY

---

# How can a glacier flow? Brief historical outline

**How could ice remain in a warm green valley without melting?**

![height:200px](fig/gletsch-1856.jpg)
<small>Rhone Glacier in 1856</small>

18th century: first theories (Rémy et al., 2006)
- 1705: **Dilation of water refreezing** (Scheuchzer)
- 1760: **Basal sliding** (Gruner)
- 1773: **Ice behaves as a “fluid”** (Bordier)

---

# 19th century: first measurements

![height:400px](fig/mercanton.jpg)

<small>Ice movement of Rhone Glacier 1874–1900 (Mercanton, 1916)</small>

➡️ Ice deformation is **faster in the center** and **slower at the sides**

---

## 20th century: from lab experiments to Glen’s flow law

![height:350px](fig/stress-strain-BuddJacka1989.png)

<small>Figure from (Budd & Jacka, 1989)</small>

- 1D: $\dot\epsilon = A(T)\,\sigma^n$, where $\dot\epsilon$ is the strain rate, and $\sigma$ is the stress.

- 3D: $\dot\varepsilon_{ij} = A(T)\,[\sigma^{(d)}_{II}]^{n-1}\,\sigma^{(d)}_{ij}$ ➡️ Ice is a **non-Newtonian fluid** <small>(J. Glen, 1958)</small>

---

# The two components of glacier ice flow

Ice is both: a **fluid** that flows/shears, and a **solid** that slides

![height:350px](fig/shearing-sliding.png)

---

# Sliding of ice under Argentière Glacier

- https://youtu.be/WkgsFiQvI_M

- https://www.youtube.com/watch?v=mHdnB_yCvLY

![height:300px](fig/argen1.JPG)

<small>Source: Luc Moreau</small>

---

# Subglacial hydrology

![height:350px](https://www.antarcticglaciers.org/wp-content/uploads/2015/06/surface_meltwater_glacier_bed.png)

- Inefficient drainage → high water pressure → high basal sliding
- Efficient drainage → low water pressure → low basal sliding

<small>Source: https://www.antarcticglaciers.org</small>

---

# Thermal and basal glacial state

![height:500px](fig/till.png)

---

# Many of these processes are coupled

![height:400px](fig/scheme.png)

... plus deformation of the lithosphere for ice fields and ice sheets

---

# Glacier response to climate

- A change in climate (ELA) changes the SMB **immediately**
- The glacier geometry adjusts **slowly**, through ice flow
- **Response time**: time needed to reach a new equilibrium after a step change in climate, roughly
$$\tau \approx \frac{H}{-b_{terminus}}$$
($H$: typical thickness, $b_{terminus}$: SMB at the terminus) ➡️ decades for small glaciers, a century or more for large ones
- Consequence: today's glaciers are **not in equilibrium** with today's climate — they will keep retreating even without further warming

You will explore this in Tutorial 1.
