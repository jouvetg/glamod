---
marp: true
theme: default
class: invert
backgroundColor: black
color: white
---

# Mini-project

---

## Objectives

- Find a glacier of interest
- Define research questions
- Sketch modelling strategies
- Learn how to report results

---

## Mains steps

---
 
### Step 1: Choose your own glacier

You may explore https://www.glims.org/maps/glims to find a glacier you would like to model. You need to collect the RGI ID (v7) and save it, as we will use it to download the data. Here are the criteria to choose your glacier:

- Prefer a well-defined mountain glacier, not an ice cap.
- The area should be between 10 km² and 200 km² (otherwise, it may be tricky computationally).  
- It is better if the glacier is somewhat dynamic (with enough slope). You may look at Google Earth for a 3D view.
- Ideally, it shows well-visible moraines from the Little Ice Age (LIA).
- Do not chose either Aletsch, or Rhone Glaciers, CH
  
Pick 3 candidate glaciers you would like to model, and comunicate to the professor (email or discord) the RGI7 ID, the name (if has one), the area, the country, by Monday 27. I will confirm the glaciers in the class on Wed 29. 

---

### Step 2: Define research questions

You now must define your research questions. The most common one relates to establishing a prognosis for the coming century based on climate RCP/SSP scenarios: (e.g., increase in air temperature or ELA):

- RCP2.6: Strong mitigation (~+1.5°C by 2100)
- RCP4.5: Moderate emissions (~+2.5°C by 2100)
- RCP8.5: High emissions (~+4.5°C by 2100)

To reach this goal, we first model past periods such as the Little Ice Age to verify that the modelling of the glacier within the next 1 to 2 centuries is consistent with observations.

If you wish, you may adress other questions such as (for instance) related to moraine formation. 

---

### Step 3: Define and perform modelling experiments

- Make sure to include the module `oggm_shop` with the RGI ID of your glacier.
- Start gently by doing a simple model run based on existing IGM examples (using simple SMB paramaters such as ELA, not a climate-based SMB yet).
- First focus on reproducing past periods, typically from the Little Ice Age (LIA, ~ 1850) to present before investigating future periods.
 
**Tips:** As we don't have surface topography for the LIA period, the trick is to start much earlier (like 1700), assuming the glacier in 1700 had the present-day topography, and then apply a transient lowering of the ELA (or the temperature), followed by an increase of the ELA (or the temperature). 

Mind that there may be a gap between the climate signal and the glacier response; this is due to glacier inertia, which varies according to the glacier volume (large glaciers usually have more inertia, and therefore longer lags).

---

### Step 4: Report your results

In your report, you must make sure to present the following information:

- Brief **rationale** of glacier modelling (why this study is important)
- Presentation of the research question(s)
- Presentation of the study site (location, characteristics, historical context)
- Brief model description of your set-up. You should explain **what** the model is doing, not **how** you are using it. It should not look like code documentation, and should be readable by someone who doesn't know about IGM.
- Presentation of the results with well-made plots (including maps, time series, cross-sections as appropriate)
- Discussion of the results and connection with the research questions
- Conclusions and potential future work

---

## Additional Tips and Common Pitfalls

---
 
### Model calibration tips

1. **Start simple**: Begin with basic parameters and gradually increase complexity. Don't try to add all features at once.

2. **Equilibrium Line Altitude (ELA) estimation**:  For present-day conditions:
   - You can estimate it from satellite imagery (snowline at end of summer)
   - Typical values range from 2500-3500m in the Alps, but vary by location
   - Historical ELA was typically 100-200m lower during the Little Ice Age 

3. **Mass balance gradients**: Typical values:
   - Ablation gradient: 0.005-0.012 m ice eq./m
   - Accumulation gradient: 0.002-0.006 m ice eq./m
 
---

### Common pitfalls to avoid

1. **Unrealistic glacier growth**: If your glacier grows indefinitely:
   - Your ELA might be too low
   - Check your accumulation parameters
 

2. **Too rapid glacier retreat**: If the glacier disappears too quickly:
   - Your ELA might be too high 
   - Check your ablation parameters

3. **Poor match with observations**:
   - Don't expect perfect agreement - models are simplified representations
   - Focus on matching general trends and patterns
   - Consider uncertainty in historical data and model parameters


---

## Suggested schedule

---
   
### Phase 1: Setup and familiarization (Week 1-2)

0. Get familiar with IGM
1. Select your glacier and collect its RGI ID
2. Download glacier data using `oggm_shop`
3. Run a simple steady-state simulation with default parameters
4. Visualize the initial glacier geometry and topography

---

### Phase 2: Historical calibration (Week 2-3)

1. Estimate historical ELA values (LIA vs present)
2. Set up a transient run from 1700 to present
3. Calibrate SMB parameters to match present-day glacier extent
4. Compare modeled results with available observations (moraines, historical maps)

---
 
### Phase 3: Future projections (Week 3-4)

1. Define climate scenarios for 2000-2100 (e.g., temperature increase)
2. Run projections under different scenarios
3. Analyze glacier response (volume loss, retreat rate, timing of disappearance)

---
 
### Phase 4: Analysis and reporting (Week 4-5)

1. Create figures
2. Write up results following the structure in Step 4
3. Discuss uncertainties and limitations
4. Draw conclusions related to your research questions

 
---

## To go further

---
   

### Extension 1: Climate driven Surface Mass Balance

Instead of using a simple SMB driven by ELA parameters, you may consider increasing the model complexity by switching to a climate-driven SMB or custom SMB.

To that aim, you may explore the IGM module `oggm_clim` and `oggm_smb`, which generate climate, and force a SMB, respectively.

---

### Extension 2: Moraine building

In case you have identified well-preserved moraines, you may switch on the particle module to try to reproduce them during the Little Ice Age to present period.

To that aim, you may explore the `particle` IGM module that permits to seed virtual particles, and transport them through the ice flow.
