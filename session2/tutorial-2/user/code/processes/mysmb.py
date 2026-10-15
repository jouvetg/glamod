# Import the most important libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math

def initialize(cfg,state):
    ice_mask = state.thk > 10
    if tf.reduce_any(ice_mask):
        state.meanela = np.quantile(state.usurf[ice_mask], 0.4)
    else:
        state.meanela = np.quantile(state.usurf, 0.4) 

def update(cfg,state):
    # perturb the ELA with a sinusoidal signal (period 200 years, amplitude 300 m) 
    ELA = (state.meanela - 300*math.sin((state.t/100)*math.pi) )
    # compute smb linear with elevation with 2 acc & abl gradients
    state.smb  = state.usurf - ELA
    state.smb *= tf.where(state.smb<0, 0.009, 0.005)  # ablation gradient 0.009, accumulation gradient 0.005
    # cap smb by 2 m/y 
    state.smb  = tf.clip_by_value(state.smb, -100, 2)
    # make sure the smb is not positive outside of the mask to prevent overflow
    if hasattr(state, "icemask"):
        state.smb  = tf.where((state.smb<0)|(state.icemask>0.5),state.smb,-10)
    
def finalize(cfg,state):
    pass
