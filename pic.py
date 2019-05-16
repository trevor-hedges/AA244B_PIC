#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 08:27:32 2019

@author: Trevor Hedges
"""

import numpy as np
import matplotlib.pyplot as plt

# Define constants
q = 1.60218E-19 # C
eps0 = 8.8542E-12 # F/m
m_i = 1.6726219E-27 # kg Assuming just protons
m_e = 9.10938356E-31 # kg
#m_e = m_i/100 # kg
K = 1.3807E-23

n0 = 5E8 # m^-3
T = 300 # K

# Boundary conditions
V_L = 0
V_R = 0

# Calculate Debye length
lambda_d = eps0*K*T/(n0*e**2)
print("lambda_d=" + str(lambda_d))

# Check # particles in debye sphere
N_d = n0*4/3*np.pi*lambda_d**3
print("N_d=" + str(N_d))

# Define sim params (make these inputs later)
delta_x = lambda_d/10
delta_t = 0.0005 # Wild guess
# deltay = deltax

n_x = 1000
n_tsteps = 100
L = delta_x*n_x
print("L="+str(L))

# Test case...
x_i_N = np.array([500, 500.5, 501.25, 505.5, 503.2, 501.2, 507.5])
x_e_N = np.array([499.5, 504, 502.6, 502.3, 506.2, 502.3, 503.9])
v_i_N = np.zeros(np.size(x_i_N))
v_e_N = np.zeros(np.size(x_e_N))

# Nondimensionalizing BCs...
V_L_N = eps0/(q*delta_x) * V_L
V_R_N = eps0/(q*delta_x) * V_R

def interpolate_charges(n_x, x_i_N, x_e_N):
    """Given charge locations, solves for charge density on 1D grid
    Using first-order interpolation
    """
        
    ind_x_i_N = np.array(np.floor(x_i_N), dtype=int)
    ind_x_e_N = np.array(np.floor(x_e_N), dtype=int)
    
    # print('ind_x_i: ' + str(ind_x_i_N))
    # print('ind_x_e: ' + str(ind_x_e_N))
    
    chg_j_i_N = (ind_x_i_N+1)-x_i_N
    chg_jp1_i_N = x_i_N-ind_x_i_N
    chg_j_e_N = -((ind_x_e_N+1)-x_e_N)
    chg_jp1_e_N = -(x_e_N-ind_x_e_N)
    # print(chg_j_i_N)
    # print(chg_jp1_i_N)
    # print(chg_j_e_N)
    # print(chg_jp1_e_N)

    # Count charges and assign to indices
    chg_N = np.zeros(n_x)
    np.add.at(chg_N, ind_x_i_N, chg_j_i_N)
    np.add.at(chg_N, ind_x_i_N+1, chg_jp1_i_N)
    np.add.at(chg_N, ind_x_e_N, chg_j_e_N)
    np.add.at(chg_N, ind_x_e_N+1, chg_jp1_e_N)
    
    return(chg_N)

def solve_potential(n_x, rho_N, V_L_N, V_R_N):
    """Given charge densities, solves for the potential on 1D grid"""
    #TODO: using what order method?
    
    # Set b vector
    b = rho_N
    
    # Add boundary conditions
    b[0] += V_L_N
    b[n_x-1] += V_R_N

    # Construct A Matrix
    FDPHI_N = -(np.diag(np.ones(n_x-1),1)+np.diag(-2*np.ones(n_x),0)+np.diag(np.ones(n_x-1),-1))

    # Perform solution    
    phi_N = np.linalg.solve(FDPHI_N,b)
    # phi is nondimensional: phi_nondim=phi*eps0/(q*delta_x^2)
    
    return(phi_N)
    
def solve_efield_from_potential(n_x, phi_N, V_L_N, V_R_N):
    """Given potential field, solves for e-field in 1D"""
    
    E_N = np.zeros(n_x);
    
    # print(np.shape(E_N))
    
    # Set interior points
    E_N[1:-1] = -(phi_N[2:] - phi_N[:-2])/2
    
    # Set boundary conditions
    E_N[0] = -(phi_N[0] - V_L_N)/2
    E_N[-1] = -(V_R_N - phi_N[-1])/2

    # Return E-field    
    return(E_N)

def integrate_motion(n_x, delta_x, delta_t, E_N, x_i_N, x_e_N, v_i_N, v_e_N):
    
    # Interpolate E-field to each particle
    # np.interp uses first-order interpolator, needed when combined w/ first-order charge-to-grid
    E_interp_i_N = np.interp(x_i_N, np.arange(n_x), E_N)
    E_interp_e_N = np.interp(x_e_N, np.arange(n_x), E_N)
    
    # Calculate new velocity
    v_i_new_N = (q*delta_t)**2/(m_i*eps0*delta_x)*E_interp_i_N + v_i_N
    v_e_new_N = -(q*delta_t)**2/(m_e*eps0*delta_x)*E_interp_e_N + v_e_N
    
    # Calculate new position
    x_i_new_N = x_i_N + v_i_new_N
    x_e_new_N = x_e_N + v_e_new_N
    
    # Return new values
    return(x_i_new_N, x_e_new_N, v_i_new_N, v_e_new_N)
    
# Tests...


# Allocate time history of particle trajectories
x_i_N_T = np.zeros([n_tsteps, np.size(x_i_N)])
x_e_N_T = np.zeros([n_tsteps, np.size(x_i_N)])

# Main Loop
for t_N in np.arange(n_tsteps):
    rho_N = interpolate_charges(n_x, x_i_N, x_e_N)
    phi_N = solve_potential(n_x, rho_N, V_L_N, V_R_N)
    E_N = solve_efield_from_potential(n_x, phi_N, V_L_N, V_R_N)
    x_i_N, x_e_N, v_i_N, v_e_N = integrate_motion(n_x, delta_x, delta_t, E_N, x_i_N, x_e_N, v_i_N, v_e_N)
    
    # Save particle positions
    x_i_N_T[t_N,:] = x_i_N
    x_e_N_T[t_N,:] = x_e_N

t = np.arange(n_tsteps)*delta_t

plt.figure()
plt.plot(t,x_i_N_T)
plt.figure()
plt.plot(t,x_e_N_T)