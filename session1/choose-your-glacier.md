# Choose your glacier

*Session 1, last hour.* Each student models **one glacier** for the mini-project, chosen from the list below (one glacier per student). The teaching team confirms the assignments at the end of Session 1.

## The candidate list

To keep things simple and fair (comparable data for everyone), we restrict the choice to the **20 largest Swiss glaciers**, excluding Aletsch and Rhone, which are used as examples in the course. All IDs are from the Randolph Glacier Inventory **version 7** (RGI v7), which is what IGM uses to download the data.

| # | Glacier | Canton | RGI v7 ID | Area (km²) | Length (km) | Elevation range (m) | Terminus (lat, lon) |
|---|---|---|---|---|---|---|---|
| 1 | Gornergletscher | VS | `RGI2000-v7.0-G-11-01225` | 56.6 | 13.0 | 2180–4510 | 45.980, 7.746 |
| 2 | Fieschergletscher | VS | `RGI2000-v7.0-G-11-02611` | 31.3 | 14.4 | 1690–4040 | 46.452, 8.136 |
| 3 | Unteraargletscher | BE | `RGI2000-v7.0-G-11-02645` | 23.6 | 12.5 | 1930–3930 | 46.567, 8.249 |
| 4 | Oberaletschgletscher | VS | `RGI2000-v7.0-G-11-02530` | 19.3 | 9.7 | 2170–3780 | 46.407, 7.999 |
| 5 | Unterer Grindelwaldgletscher (Fieschergletscher BE) | BE | `RGI2000-v7.0-G-11-01581` | 18.8 | 9.1 | 1290–4060 | 46.603, 8.053 |
| 6 | Glacier de Corbassière | VS | `RGI2000-v7.0-G-11-00823` | 16.7 | 9.0 | 2230–4270 | 46.010, 7.292 |
| 7 | Vadret da Morteratsch | GR | `RGI2000-v7.0-G-11-02216` | 15.8 | 7.1 | 2000–3960 | 46.429, 9.935 |
| 8 | Zmuttgletscher | VS | `RGI2000-v7.0-G-11-00978` | 15.4 | 7.8 | 2240–3950 | 46.002, 7.658 |
| 9 | Triftgletscher | BE | `RGI2000-v7.0-G-11-01728` | 14.8 | 6.6 | 1650–3370 | 46.691, 8.358 |
| 10 | Findelengletscher | VS | `RGI2000-v7.0-G-11-01189` | 14.3 | 7.4 | 2490–3770 | 46.010, 7.818 |
| 11 | Glacier de Zinal | VS | `RGI2000-v7.0-G-11-00922` | 14.0 | 7.3 | 2040–3910 | 46.083, 7.632 |
| 12 | Kanderfirn | BE | `RGI2000-v7.0-G-11-01457` | 13.2 | 6.5 | 2290–3190 | 46.467, 7.762 |
| 13 | Hüfifirn | UR | `RGI2000-v7.0-G-11-01870` | 12.1 | 7.1 | 1640–3220 | 46.807, 8.812 |
| 14 | Glacier d'Otemma | VS | `RGI2000-v7.0-G-11-00859` | 11.8 | 8.7 | 2450–3760 | 45.934, 7.413 |
| 15 | Gauligletscher | BE | `RGI2000-v7.0-G-11-01647` | 9.4 | 6.1 | 2140–3570 | 46.610, 8.211 |
| 16 | Allalingletscher | VS | `RGI2000-v7.0-G-11-01182` | 9.3 | 6.0 | 2660–4180 | 46.049, 7.943 |
| 17 | Langgletscher | VS | `RGI2000-v7.0-G-11-01515` | 9.3 | 6.4 | 2060–3850 | 46.449, 7.908 |
| 18 | Oberer Grindelwaldgletscher | BE | `RGI2000-v7.0-G-11-02652` | 9.0 | 5.9 | 1370–3650 | 46.627, 8.082 |
| 19 | Feegletscher | VS | `RGI2000-v7.0-G-11-01161` | 8.1 | 5.4 | 2460–4030 | 46.088, 7.903 |
| 20 | Steingletscher | BE | `RGI2000-v7.0-G-11-01755` | 8.1 | 6.5 | 1930–3440 | 46.722, 8.434 |

*Source: RGI v7 attributes (outlines ~2000). Today's glaciers are smaller! The same table is available as [candidate-glaciers.csv](candidate-glaciers.csv).*

<!-- TEACHER NOTES
 - List = 20 largest RGI7 region-11 glaciers inside Switzerland, minus Aletsch (G-11-02596), Rhone (G-11-01706)
   Mont Miné (G-11-00907) and Ferpècle (G-11-00910), removed on purpose. Next in line if more students:
   Roseg G-11-02195 (7.8), Riedgletscher G-11-01174 (7.6), Mittelaletsch G-11-01522 (7.5). Plaine Morte (7.7) skipped: flat plateau glacier.
 - #11 is named "Grand Cornier-E-I" in RGI7 but its terminus is at the Zinal glacier tongue -> check outline on GLIMS.
 - #5 is named "FIESCHERGLETSCHER BE" in RGI7; terminus at 1290 m = Unterer Grindelwaldgletscher.
 - Last year's CSV had G-11-02216 for both Tschierva and Morteratsch: 02216 = Morteratsch (Tschierva = 02196).
 - Test every ID with IGM 3.2 on Renku before the course (setup/renku.md).
-->

## Resources to explore your glacier

- **[map.geo.admin.ch](https://map.geo.admin.ch)** — *the* key resource for Swiss glaciers. Search for your glacier and use the layers:
  - *Glacier inventory* (outlines through time) and *glacier thickness* (ice thickness map, Grab et al. 2021),
  - the **time slider** ("Journey through time") on the national maps, back to the Siegfried maps (~1870–1940) and the Dufour map: you can see the glacier extent during the late Little Ice Age (LIA),
  - the 3D view, to look at slopes, tributaries and moraines.
- **[GLIMS viewer](https://www.glims.org/maps/glims)** — glacier outlines and RGI IDs (switch on the RGI 7 layer).
- **[IGE velocity viewer](https://ige-vis.univ-grenoble-alpes.fr)** — observed surface ice speeds.
- **[IceBoost web app](https://nmaffe.github.io/iceboost_webapp/)** — modelled ice thickness for glaciers worldwide.
- **[GLAMOS](https://www.glamos.ch)** — Swiss glacier monitoring: length changes since the 19th century, mass balance, volume changes.

## What to do during this hour

1. **Explore** 2–3 candidates with the resources above.
2. **Choose** your glacier and put your name next to it in the shared sign-up sheet (link given in class). One glacier per student.
3. **Write down** what you will need for the introduction of your report:
   - name, RGI ID, location, area, length, elevation range,
   - LIA extent: where is the LIA moraine, and how far below today's front is it? (Siegfried map, swisstopo time slider)
   - observations: length-change record (GLAMOS), surface velocities (IGE), ice thickness (swisstopo / IceBoost),
   - anything special (debris cover, tributaries, proglacial lake, hazards, history, ...).

➡️ In Session 2, you will download the data for your glacier with IGM and run your first simulations.
