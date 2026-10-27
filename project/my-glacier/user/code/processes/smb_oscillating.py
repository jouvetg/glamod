# User module (exploration "response time"): ELA-based SMB with an ELA oscillating in time.
# Like any IGM module, it has 3 functions that read and modify the shared `state`:
# initialize (called once), update (at each time step) and finalize (at the end).
# Its parameters are in user/conf/processes/smb_oscillating.yaml.

import math
import numpy as np
import tensorflow as tf

def initialize(cfg, state):
    p = cfg.processes.smb_oscillating
    if p.ela_mean is None:
        # default mean ELA: 40 % quantile of today's glacier surface elevation
        ice_mask = state.thk > 10
        z = state.usurf[ice_mask] if tf.reduce_any(ice_mask) else state.usurf
        state.ela_mean = float(np.quantile(z, 0.4))
    else:
        state.ela_mean = float(p.ela_mean)
    print(f"smb_oscillating: mean ELA = {state.ela_mean:.0f} m, "
          f"amplitude = {p.amplitude} m, period = {p.period} yr")

def update(cfg, state):
    p = cfg.processes.smb_oscillating
    # ELA oscillating around its mean value (starts at the mean, then goes DOWN = glacier grows)
    t = float(state.t.numpy()) - cfg.processes.time.start
    state.ela = state.ela_mean - p.amplitude * math.sin(2 * math.pi * t / p.period)
    # SMB linear in elevation, with ablation / accumulation gradients, capped at accmax
    smb = state.usurf - state.ela
    smb *= tf.where(smb < 0, p.gradabl, p.gradacc)
    smb = tf.clip_by_value(smb, -100.0, p.accmax)
    # no accumulation outside the glacier mask (prevents ice growing on surrounding peaks)
    if hasattr(state, "icemask"):
        smb = tf.where((smb < 0) | (state.icemask > 0.5), smb, -10.0)
    state.smb = smb

def finalize(cfg, state):
    pass
