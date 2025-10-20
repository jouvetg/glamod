---
marp: true
theme: default
class: invert
backgroundColor: black
color: white
---

# Introduction

Guillaume Jouvet
   
![height:200px](fig/rhone-mod-lq.png)

---

# Course objectives

- Provide basics of glacier processes
- Introduce numerical methods to model glacier evolution
- Perform pratical numerical experiments with IGM

# Ressources

- If there is one ressource to remember is the awesome website : https://www.antarcticglaciers.org that can allow you to deepen the processes.

- To dive more into the mechanics and the maths, I recommend the "Karthaus Summer School Lecture Notes": 
https://link.springer.com/book/10.1007/978-3-030-42584-5

---

# Content of this course

- I> Glacier processes
- II> Mathematical Modelling
- III> Numerical Modelling
- IV> Application to paleo and present-day glaciers 
- V> Analogue Modelling
 
---

# I > Glacier processes  

---

# Main glacial processes governing glacier evol.

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

# Observing glacial motion

There are a number of timelapse cameras that have captured glacial motion :

- https://www.youtube.com/watch?v=TWGR6FxFlt8

- https://www.youtube.com/watch?v=he5QzhE7_g4&t=55s

- https://www.youtube.com/watch?v=iaULB6O-JNY 

---

# How can a glacier flow?  Brief historical outlines

**How could ice remain in a warm green valley without melting?**

![height:200px](fig/gletsch-1856.jpg)  
<small>Rhone Glacier in 1856</small>

18th century: 1st theories (Rémy et al., 2006)  
- 1705: **Dilation of water refreezing** (Scheuchzer)  
- 1760: **Basal sliding** (Gruner)  
- 1773: **Ice behaves as a “fluid”** (Bordier)  

---

# 19th century: 1st measurements  


![height:400px](fig/mercanton.jpg)  

<small>Ice movement of Rhone Glacier 1874–1900 (Mercanton, 1916)</small>

➡️ Ice deformation is **faster in the center** and **slower at the sides**


---

# 20th century: From lab exp. to Glen’s flow law  

![height:350px](fig/stress-strain-BuddJacka1989.png)

<small>Figure from (Budd & Jacka, 1989)</small>

- 1D:  $\dot\epsilon = A(T)\,\sigma^n$, where $\dot\epsilon$ is the strain rate, and $\sigma$  is the stress.

- 3D: $\dot\varepsilon_{ij} = A(T)\,[\sigma^{(d)}_{II}]^{n-1}\,\sigma^{(d)}_{ij}$  ➡️ Ice is a **non-Newtonian fluid**  <small>(J. Glen, 1958)</small>

---

# The two components of glacier ice flow

ICE is both: a **fluid** that flow/shear, and a **solid** that slides

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

- Inefficient drainage → high pressure → high basal sliding  
- Efficient drainage → low pressure → low basal sliding  

<small>Source: https://www.antarcticglaciers.org</small>

---

# Thermic and basal glacial state

![height:500px](fig/till.png)

---

# Number of these processes are coupled 

![height:400px](fig/scheme.png)

... and deformation of lithosphere for ice fields and ice sheets

---

# II> Mathematical Modelling

---
 
# Glen's flow law

<div style="display:flex;align-items:center;">
 
![height:350px](fig/stress-strain-BuddJacka1989.png)

- 1D:  $\dot\epsilon = A(T)\,\sigma^n$, where $\dot\epsilon$ is the strain rate, and $\sigma$  is the stress.

- 3D: $\dot\varepsilon_{ij} = A(T)\,[\sigma^{(d)}_{II}]^{n-1}\,\sigma^{(d)}_{ij}$  ➡️ Ice = non-Newtonian fluid  <small>(J. Glen, 1958), n=3</small>

---

# Ice dynamics equations & boundary conditions

![height:450px](fig/gl-equations.png)

Note that the ice flow speed is independent of time!


---

# Shallow Ice models (since glaciers are shallow)

| | |
|---|---|
| ![height:300px](fig/pic2.jpg) | ![height:300px](fig/pic1.jpg) |
| *source: G. Kappenberger* | *source: NASA* |
| Mountain glacier | Ice sheet |
| aspect ratio $\sim 1/10$ | aspect ratio $\sim 1/1000$ |

<br>
 
---

# Shallow Ice Approximation (SIA)

Most simple ice flow model, velocity given by formula ($n=3$):

$$
u(z) = \underbrace{\frac{-2A}{n+1}\left(\rho g \frac{\partial s}{\partial x}\right)^n
\left(H^{n+1}-(H-z)^{n+1}\right)}_{\text{deformation velocity}}
+ \underbrace{u_b}_{\text{sliding velocity}} \qquad (1)
$$

SIA is obtained by neglecting $\mathcal{O}(\epsilon^k)$,  
where $\epsilon$ is the aspect ratio.

**Valid for**: thin ice, inland ice sheets, wide glaciers.  
**Not valid for**: thick ice, ice domes, narrow glaciers, fast sliding.

---

# Shallow Ice Approximation (SIA)

![height:500px](fig/SIA-profile.png) 

---

# 2 (simplified) shallow ice models

- **SIA** (Shallow Ice Approximation) -> suitable for pure **Shearing**
- **SSA** (Shallow Shelf Approximation) -> suitable for pure **Sliding**


![height:450px](fig/sia-vs-ssa.png)

---

# Dynamics of marine ice sheets

![height:500px](fig/scheme-ice-shelf.png)

---

# Dynamics of the Antartica Ice Sheet

![height:350px](https://svs.gsfc.nasa.gov/vis/a000000/a003800/a003849/antarctica_flows_1_00120_1024x576.jpg)

*Ice flow field in Antarctica, the pink areas display the zone for the ice is the fastest over ice shelves (floating ice). Check a the [NASA website](https://svs.gsfc.nasa.gov/3849/) for animations.* Source: NASA.

---

# Connecting ice flow and mass conservation

Mass conservation: rate of change of thickness = surface mass balance (SMB)

![height:350px](fig/simple_glacierA.png)

$$\frac{dh}{dt}(x,t) = \frac{\partial h}{\partial t} + \frac{\partial}{\partial x} \left( \int_b^s u \, dz \right) = b \qquad (2) $$

---

# Ice sheet evolution equation

Combine SIA (1) with $n=3$, and no sliding with mass conservation equation (2) leads to a nonlinear diffusive equation (given here in 1D) that predicts the evolution of the geometry of an ice sheet:

$$ \frac{\partial h}{\partial t} = \frac{\partial}{\partial x}\left(D(h)\frac{\partial z}{\partial x}\right) + b(z), \qquad (3) $$

where D(h) is the dynamic diffusivity of the ice defined by

$$ D(h) = \frac{2A}{5} (\rho g)^3 h^5 \left(\frac{\partial z}{\partial x}\right)^2 \qquad (4) $$
 
---
 
# III > Numerical modelling  

---

# Need for a numerical glacier evolution model

Since the above equations are to complex to be solved analytically, we need a **numerical model** to approximate them.

 
  - Link to the course on [numerical modelling](https://jouvetg.github.io/modnum/).

  - 5 min video on glacier modelling : https://youtu.be/eJNIr_0zOyk

<figure style="text-align: center;"> 
    <img src="fig/rhone-mod.png" height="200" />  
</figure>
<sub> Numerically modelled retreat of Rhone Glacier from 1874 to 2100 (Jouvet and al., JCP, 2019)</sub>

---

# Glacier must be meshed/discretized ...

to resolve the equations numerically.

![height:400px](fig/MESH.png)

In practise, it is common to work with simplified (shallow) models.

---

# Transient glacier evolution computation

![height:600px](fig/algo.gif)

---

# Combining ice flow and mass balance models  

The above algorithm updates the glacier surface at each time step accounting for both: ice flow and surface mass balance

![height:400px](fig/velocity-smb-rhone.png)  


--- 

# Numerical models

- The SIA-based equations can solved in few lines of codes, see the last courses / exercices sheets of the course on [numerical modelling](https://jouvetg.github.io/modnum/), you may deepen this aspect with the [Karthaus Summer School Lecture Notes](https://link.springer.com/book/10.1007/978-3-030-42584-5)

- There are a number of open models developped by glaciology modeller community that solves ice flow equations with different level complexity such as [Elmer/Ice](https://elmerice.elmerfem.org/), [PISM](https://www.pism.io/), ISSM, CISM, OGGM, IcePack, ...

- Here we will use the "Instructed Glacier Model" ([IGM](https://igm-model.org/) developed at UNIL), which is a python-based glacier evolution model that uses Machine Learning (ML) and Graphics Processing Units (GPU) to acceleratehe code computations.

---

# IV > Present-day and paleo glacier applications  

---

# Modelling of the evolution of Aletsch Glacier

![height:300px](fig/aletsch.jpg)

<small>Key numbers: ~ 20 km long – 85 km² – 13 km³ </small>  

➡️ 1880–2010 : Model validation <small>(Jouvet and al., JOG 2010)</small>  
➡️ 2010–2100 : Multiple climate scenarios <small>(Jouvet & Huss, JOG 2019)</small>  


Check at the [simulation page](https://jouvetg.github.io/the-aletsch-glacier-module)


---

# Solving a cold case with glacier modelling!

![height:250px](fig/climbers2.jpg)
 
➡️ 1226: 4 mens vanish on the Great Aletsch Glacier
➡️ 2012: The remains of 3 of them ar found ~ 10 km downstream.

What caused the fate of the mountainers? 
Can glacier modelling help to estimate the vanishing place?

Want to know more ? Check at the [short film](https://youtu.be/cyKb-P3mwDk).


---
 
# Last Glacial Maximum in the Switzerland

![height:450px](fig/AIF-CH.png)

Can we reproduce this with a glacier model ?

---
 
## Paleo glacier modelling in the Alps with IGM

![height:370px](fig/AIF-leger-lq.png)
 
<sub>(Leger, Jouvet and al., Nature Com., 2025)   </sub>

Modelling last ice age glacier evolution in the Alps: https://youtu.be/IbLOFh3U9gI 
 
---

# Ice flow modelling for landscape evolution

... to model the alternance of ice flow-driven glacial (U) and fluvial (V) erosion.

![height:370px](fig/cordonnier_lq.png)

 
<sub>(Cordonnier, Jouvet et al., SIGGRAPH, 2023)</sub>

Simulation: https://youtu.be/xfk_J4VhdWA

---

# V > Analogue modelling

---

# The "glacier goo" experiment

https://uzh.mediaspace.cast.switch.ch/channel/Glacier-Goo-Experimente

![height:400px](fig/glacier-goo.png)

Credit: G. Vieli, A. Vieli, A. Linsbauer (UZH, 3G)
