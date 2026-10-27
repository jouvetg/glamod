#!/bin/bash
# EXPLORATION "ice-flow parameters": surface speeds of today's glacier for a grid of
# ice softness A (arrhenius) x basal sliding (tau_ref). Short 1-year runs (~15 s each, 12 runs).
# In a terminal, from your copy ~/work/my-glacier/:   bash run_sensitivity.sh
# Then compare with the observed speeds in analysis.ipynb (compare_velocity_misfit).
ARRHENIUS="50 78 150"      # MPa^-3 a^-1 (default 78): higher = softer ice
TAU_REF="0.1 0.215 0.4 0.8"  # MPa (default 0.215): lower = more sliding

for a in $ARRHENIUS; do
  for t in $TAU_REF; do
    echo "=== sens_A${a}_tau${t}"
    igm_run +experiment=params_future processes.time.end=2021 \
      processes.iceflow.physics.viscosity.arrhenius=$a \
      processes.iceflow.physics.sliding.tau_ref=$t \
      hydra.run.dir=outputs/sens_A${a}_tau${t} || exit 1
  done
done
echo "Done: outputs/sens_*  ->  compare_velocity_misfit in analysis.ipynb"
