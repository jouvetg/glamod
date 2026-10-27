#!/bin/bash
# FUTURE scenarios 2020-2100 for my glacier.
# 1. Set the 3 numbers below (from the past calibration and plot_climate in analysis.ipynb).
# 2. In a terminal, from project/my-glacier/:   bash run_future.sh
# Each scenario runs params_future.yaml with a different 2100 delta_temp, into outputs/future_<name>/.
# (The values actually used are recorded in outputs/future_<name>/.hydra/overrides.yaml.)

MELT_ENHANCER=1.0   # <- calibrated with the past run
DT_2020=1.3         # <- present-day offset vs 1960-1990 (printed by plot_climate)
DT_8110=0.6         # <- 1981-2010 offset vs 1960-1990 (printed by plot_climate)
COARSENING=2        # <- grid coarsening: 2 for quick tests, 1 for the final runs of your report

# name:delta_temp in 2100 (vs 1960-1990) = 1981-2010 offset + CH2018 warming (see Lecture 2, Part A, Step 3)
add() { python3 -c "print(round($1 + $2, 2))"; }
SCENARIOS="committed:$DT_2020 rcp26:$(add $DT_8110 1.2) rcp45:$(add $DT_8110 2.6) rcp85:$(add $DT_8110 4.3)"

for s in $SCENARIOS; do
  name=${s%%:*}; dt2100=${s##*:}
  echo "=== future_$name: delta_temp 2020 = $DT_2020, 2100 = $dt2100 (vs 1960-1990), melt_enhancer = $MELT_ENHANCER, coarsening = $COARSENING"
  igm_run +experiment=params_future \
    "processes.climate.oggm.clim_trend_array=[[time,delta_temp,prec_scal],[2020,$DT_2020,1.0],[2100,$dt2100,1.0]]" \
    processes.smb.oggm.melt_enhancer=$MELT_ENHANCER \
    inputs.local.coarsening.ratio=$COARSENING \
    hydra.run.dir=outputs/future_$name || exit 1
done
echo "Done: outputs/future_*  ->  plot them in analysis.ipynb"
