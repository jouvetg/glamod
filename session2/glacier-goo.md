---
marp: true
theme: default
class: invert
backgroundColor: black
color: white
---

# Break: glacier goo

## Ice flow in the lab

![height:300px](fig/glacier-goo.png)

9 short films (≈ 1 min each) by A. Linsbauer, G. Leysinger Vieli, A. Vieli (UZH, 3G)
Channel: https://uzh.mediaspace.cast.switch.ch/channel/Glacier-Goo-Experimente

---

# Why a "goo"?

- Glacier ice behaves like a very **viscous, non-Newtonian fluid** (Glen's law, Lecture 1b) — over years, not seconds
- "Glacier goo" (a polymer slime) is also a viscous fluid, but flows in **minutes**
- Same physics as in IGM: gravity drives the flow, the bed resists it; mass is conserved
- Watch for: **where it flows fast or slow, how it reacts to the bed, how surface markers deform**

<small>Background (in German): Linsbauer, Leysinger Vieli, Vieli & Kubik (2025), *Wie fließt ein Gletscher? Das dynamische Fließen der Gletscher experimentell erfahren*, Geographie Heute 372, 28–37.</small>

---

# 1 — Building a glacier: flow and advance

*01 Gletscher: Aufbau, Fliessen, Vorstoss* (1:05)
https://uzh.mediaspace.cast.switch.ch/media/01_Gletscher_Aufbau_Fliessen_Vorstoss/0_foa6iyh6

- Goo is added at the top ("accumulation"): the glacier thickens, steepens, then **flows and advances** down-valley
- ➡️ Mass conservation + ice flow: the same loop as in Tutorial 1 and IGM
- ➡️ A front advances only once enough mass has been transported down: **response time**

---

# 2 — Flow speeds within a glacier

*02 Fliessgeschwindigkeiten* (1:07)
https://uzh.mediaspace.cast.switch.ch/media/02_Fliessgeschwindigkeiten/0_1g1b0yhw

- A line of markers across the surface bends downstream
- **Fast in the centre, slow at the margins** — like Mercanton's stakes on the Rhone Glacier (1874–1900, Lecture 1a)
- ➡️ Lateral drag and thickness control speed ($u \propto H^{n+1}$ in the SIA)

---

# 3 — Confluence and medial moraines

*03 Konfluenz, Mittelmoränen* (1:02)
https://uzh.mediaspace.cast.switch.ch/media/03_Konfluenz_Mittelmoraenen/0_zufmrrzh

- Two tributaries merge; colored goo stays side by side, **without mixing**
- Debris from the ridge between them forms a **medial moraine** (e.g. Aletsch, Gorner)
- ➡️ Laminar flow: particle tracking in IGM (`particles` module) reproduces this

---

# 4 — Rockfall and rock avalanches

*04 Steinschlag, Felsstürze* (1:27)
https://uzh.mediaspace.cast.switch.ch/media/04_Steinschlag_Felsstuerze/0_lrcqve3j

- Debris falls on the surface and is **carried down-glacier** with the ice
- Debris in the accumulation area is buried and re-emerges in the ablation area
- ➡️ Same logic as the Ebener brothers' remains (Lecture 1b): buried up-glacier, emerging at the tongue decades later

---

# 5 — Flow patterns

*05 Fliessformen* (1:30)
https://uzh.mediaspace.cast.switch.ch/media/05_Fliessformen/0_oisqx6mm

- Initially straight surface patterns are folded and stretched into **arcs and loops**
- The shape records the velocity field over time
- ➡️ Compare with moraine loops and folded debris bands on real glaciers (aerial images, map.geo.admin.ch)

---

# 6 — Flow over a bedrock step

*06 Fliessen Steilstufe* (0:46)
https://uzh.mediaspace.cast.switch.ch/media/06_Fliessen_Steilstufe/0_5h58kq70

- Over a steep step: the goo **thins and accelerates**, then thickens and slows below
- Flux is conserved: $q = \bar u\,H$ ≈ constant → when $\bar u$ goes up, $H$ goes down
- ➡️ Icefalls; this is why the bed topography matters so much in IGM

---

# 7 — Extension and compression

*07 Dehnung, Stauchung* (0:47)
https://uzh.mediaspace.cast.switch.ch/media/07_Dehnung_Stauchung/0_ururz8au

- Where the flow accelerates: **extension** → cracks open (crevasses)
- Where it decelerates: **compression** → bulges, ogives, thrusting near the front
- ➡️ Strain rates = spatial gradients of velocity: the quantity in Glen's law

---

# 8 — Rock glacier flow

*08 Blockgletscher Fliessen* (0:37)
https://uzh.mediaspace.cast.switch.ch/media/08_Blockgletscher_Fliessen/0_xgaiy2vm

- Goo under a transparent sheet: a debris-covered, ice-rich body (permafrost) creeping slowly
- Lobes and ridges form at the surface
- ➡️ Same creep physics, but much slower (cm–m/yr) and with a debris mantle

---

# 9 — Ice creep and basal sliding

*09 Eiskriechen, Gleiten* (0:39)
https://uzh.mediaspace.cast.switch.ch/media/09_Eiskriechen_Gleiten/0_tmtityxc

- Transparent goo in a Plexiglas channel, side view: the **velocity profile** with depth
- Pure creep (no slip at the bed) vs. **sliding** (the base moves too)
- ➡️ The two components of ice flow (Lecture 1a); in IGM: `arrhenius` (creep) and `tau_ref` (sliding)

---

# Take-home

| Goo experiment | Glacier process | In the model |
|---|---|---|
| 1 Build-up, advance | Mass balance + flow | SMB + `thk` (mass conservation) |
| 2, 6, 9 Speeds, step, creep/sliding | Ice dynamics | `iceflow` (Glen's law, sliding) |
| 3, 4, 5 Moraines, debris, folds | Transport | `particles` |
| 7 Extension/compression | Strain, crevasses | Strain rates from the velocity field |
| 8 Rock glacier | Permafrost creep | (not in this course) |

Back to the mini-project!
