---
marp: true
theme: default
class: invert
backgroundColor: black
color: white
---

<style>
table { font-size: 0.75em; }
</style>

# Choose your glacier

## Session 1 — Hour 4

Each student models **one glacier** for the mini-project, chosen from the list (one glacier per student).

The teaching team confirms the assignments at the end of Session 1.

---

# The candidate list

To keep things simple and fair (comparable data for everyone):

- the **20 largest Swiss glaciers**
- excluding Aletsch and Rhone, used as examples in the course
- IDs from the Randolph Glacier Inventory **version 7** (RGI v7): what IGM uses to download the data

*Source: RGI v7 attributes (outlines ~2000). Today's glaciers are smaller!*
Full table (elevation range, terminus position): [candidate-glaciers.csv](candidate-glaciers.csv)

---

# Candidates 1–10

| # | Glacier | Canton | RGI v7 ID | Area (km²) | Length (km) |
|---|---|---|---|---|---|
| 1 | Gornergletscher | VS | `RGI2000-v7.0-G-11-01225` | 56.6 | 13.0 |
| 2 | Fieschergletscher | VS | `RGI2000-v7.0-G-11-02611` | 31.3 | 14.4 |
| 3 | Unteraargletscher | BE | `RGI2000-v7.0-G-11-02645` | 23.6 | 12.5 |
| 4 | Oberaletschgletscher | VS | `RGI2000-v7.0-G-11-02530` | 19.3 | 9.7 |
| 5 | Unt. Grindelwaldgletscher | BE | `RGI2000-v7.0-G-11-01581` | 18.8 | 9.1 |
| 6 | Corbassière | VS | `RGI2000-v7.0-G-11-00823` | 16.7 | 9.0 |
| 7 | Morteratsch | GR | `RGI2000-v7.0-G-11-02216` | 15.8 | 7.1 |
| 8 | Zmuttgletscher | VS | `RGI2000-v7.0-G-11-00978` | 15.4 | 7.8 |
| 9 | Triftgletscher | BE | `RGI2000-v7.0-G-11-01728` | 14.8 | 6.6 |
| 10 | Findelengletscher | VS | `RGI2000-v7.0-G-11-01189` | 14.3 | 7.4 |

---

# Candidates 11–20

| # | Glacier | Canton | RGI v7 ID | Area (km²) | Length (km) |
|---|---|---|---|---|---|
| 11 | Zinal | VS | `RGI2000-v7.0-G-11-00922` | 14.0 | 7.3 |
| 12 | Kanderfirn | BE | `RGI2000-v7.0-G-11-01457` | 13.2 | 6.5 |
| 13 | Hüfifirn | UR | `RGI2000-v7.0-G-11-01870` | 12.1 | 7.1 |
| 14 | Otemma | VS | `RGI2000-v7.0-G-11-00859` | 11.8 | 8.7 |
| 15 | Gauligletscher | BE | `RGI2000-v7.0-G-11-01647` | 9.4 | 6.1 |
| 16 | Allalingletscher | VS | `RGI2000-v7.0-G-11-01182` | 9.3 | 6.0 |
| 17 | Langgletscher | VS | `RGI2000-v7.0-G-11-01515` | 9.3 | 6.4 |
| 18 | Ob. Grindelwaldgletscher | BE | `RGI2000-v7.0-G-11-02652` | 9.0 | 5.9 |
| 19 | Feegletscher | VS | `RGI2000-v7.0-G-11-01161` | 8.1 | 5.4 |
| 20 | Steingletscher | BE | `RGI2000-v7.0-G-11-01755` | 8.1 | 6.5 |

<!-- TEACHER NOTES
 - List = 20 largest RGI7 region-11 glaciers inside Switzerland, minus Aletsch (G-11-02596), Rhone (G-11-01706)
   Mont Miné (G-11-00907) and Ferpècle (G-11-00910), removed on purpose. Next in line if more students:
   Roseg G-11-02195 (7.8), Riedgletscher G-11-01174 (7.6), Mittelaletsch G-11-01522 (7.5). Plaine Morte (7.7) skipped: flat plateau glacier.
 - #11 is named "Grand Cornier-E-I" in RGI7 but its terminus is at the Zinal glacier tongue -> check outline on GLIMS.
 - #5 is named "FIESCHERGLETSCHER BE" in RGI7; terminus at 1290 m = Unterer Grindelwaldgletscher.
 - Last year's CSV had G-11-02216 for both Tschierva and Morteratsch: 02216 = Morteratsch (Tschierva = 02196).
 - Test every ID with IGM 3.2 on Renku before the course (setup/renku.md).
-->

---

# Resources to explore your glacier

- **[map.geo.admin.ch](https://map.geo.admin.ch)**: *the* key resource for Swiss glaciers
  - *Glacier inventory* (outlines through time) and *glacier thickness* (Grab et al. 2021)
  - **time slider** ("Journey through time"): Siegfried (~1870–1940) and Dufour maps ➡️ extent at the end of the Little Ice Age (LIA)
  - 3D view: slopes, tributaries, moraines
- **[GLIMS viewer](https://www.glims.org/maps/glims)**: outlines and RGI IDs (RGI 7 layer)
- **[IGE velocity viewer](https://ige-vis.univ-grenoble-alpes.fr)**: observed surface ice speeds
- **[IceBoost web app](https://nmaffe.github.io/iceboost_webapp/)**: modelled ice thickness
- **[GLAMOS](https://www.glamos.ch)**: Swiss glacier monitoring (length changes since the 19th century, mass balance)

---

# What to do during this hour

1. **Explore** 2–3 candidates with the resources above
2. **Choose** your glacier: your name in the shared sign-up sheet (one glacier per student)
3. **Write down** for the introduction of your report:
   - name, RGI ID, location, area, length, elevation range
   - LIA extent: where is the LIA moraine? (Siegfried map, time slider)
   - observations (GLAMOS, IGE, swisstopo), anything special (debris, lakes, hazards...)

➡️ Session 2: you download the data for your glacier with IGM and run your first simulations
