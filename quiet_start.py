#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 20:19:34 2019

@author: linda
"""

import numpy as np
from scipy.special import erfinv

def maxwellian(N, vd, vth):
    # Need to use this to calculate q and m magnitudes
    # N is number of particles, vd is drift velocity, vth is thermal vel
    
    # Subdivide into 3 vthermal
    print(N)
    
    i = np.arange(-1+1/N,1,2/N)
    vx_norm = erfinv(i) # v/vth

    vx = vth*vx_norm+vd
    
    #print("initial velocities... ")
    #print(vx)
    
    # Scramble
    np.random.shuffle(vx)
    
    return(vx)

    #rx = np.random.rand()
    
    #np.exp(-(vx-vd)**2/(2*vth**2))