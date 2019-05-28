#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 08:27:32 2019

@author: Trevor Hedges
"""

import numpy as np
import matplotlib.pyplot as plt

import quiet_start

# Define physical constants

q_phys = 1.60218E-19 # C
m_i_phys = 1.6726219E-27 # kg Assuming just protons
m_e_phys = 9.10938356E-31
#m_e_phys = m_i/10 # kg

eps0 = 8.8542E-12 # F/m
K = 1.3807E-23


# "True" plasma density to simulate
n0 = 1E8

q_to_m_i = q_phys/m_i_phys
q_to_m_e = q_phys/m_e_phys



# Define simulation constants
mi_me_ratio = 100

macroparticle_num = 5000
q = macroparticle_num*q_phys
m_e = q/q_to_m_e
m_i = mi_me_ratio*m_e

print('me=' + str(m_e) + " kg")
print('mi=' + str(m_i) + " kg")

# Boundary conditions
V_L = 0
V_R = 0

# Calculate electron Debye length
T = 100;
lambda_d = np.sqrt(eps0*K*T/(n0*q_phys**2))
print("lambda_d=" + str(lambda_d) + " m")

# Calculate electron plasma frequency
omega_p = np.sqrt(n0*q_phys**2/(eps0*m_e_phys))
print("omega_p=" + str(omega_p) + " rad/s")

# Check # particles in debye sphere
N_d = n0*4/3*np.pi*lambda_d**3
print("N_d=" + str(N_d))

# Define sim params (make these inputs later)
delta_x = lambda_d/10
delta_t = 0.02/omega_p
print('delta_x=' + str(delta_x) + ' m')
print('delta_t=' + str(delta_t) + ' s')
# deltay = deltax

n0_sim = n0/macroparticle_num # m^-1
Ti = 100 # K
Te = 100 # K
vthi = np.sqrt(K*Ti/m_i_phys)*delta_t/delta_x
vthe = np.sqrt(K*Te/m_e_phys)*delta_t/delta_x

print(vthi)
print(vthe)

# NP = 500 # Number of one species
n_x = 1001
n_tsteps = 500
t_tot = delta_t*n_tsteps
plasma_cycles = (omega_p/(2*np.pi))*t_tot
print("Total time to be simulated: " + str(t_tot) + " seconds; " + str(plasma_cycles) + " plasma oscillations")

# Test case initial conditions...
chunk_start = 450
chunk_end = 550
chunk_len = chunk_end-chunk_start
NP = int(np.round(n0_sim*delta_x*chunk_len))

print('number of each species=' + str(NP))
L = delta_x*chunk_len
print("L="+str(L)+" m")

# Check plasma criteria
if lambda_d > L:
    print("WARNING: Plasma criteria lambda_d << L is DEFINITELY not met.")
elif lambda_d > L/10:
    print("Warning: Plasma criteria lambda_d << L is in danger of not being met.")
else:
    print("Plasma criteria lambda_d << L is met!")
if N_d <= 1:
    print("WARNING: Plasma criteria N_debye >> 1 is DEFINITELY not met.")
elif N_d <= 10:
    print("Warning: Plasma criteria N_debye >> 1 is in danger of not being met.")
else:
    print("Plasma criteria N_debye >> 1 is met!")
# When collisions added, need to add this criteria

if NP <= 100000:
    safe_to_run = True
else:
    safe_to_run = False
    print('WAY TOO MANY PARTICLES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')



def interpolate_charges(n_x, x_i_N, x_e_N, q_i_N, q_e_N):
    """Given charge locations, solves for charge density on 1D grid
    Using first-order interpolation
    """
    
    
    ind_x_i_N = np.array(np.floor(x_i_N), dtype=int)
    ind_x_e_N = np.array(np.floor(x_e_N), dtype=int)
    
    # print('ind_x_i: ' + str(ind_x_i_N))
    # print('ind_x_e: ' + str(ind_x_e_N))
    
    chg_j_i_N = q_i_N*((ind_x_i_N+1)-x_i_N)
    chg_jp1_i_N = q_i_N*(x_i_N-ind_x_i_N)
    chg_j_e_N = -q_e_N*((ind_x_e_N+1)-x_e_N)
    chg_jp1_e_N = -q_e_N*(x_e_N-ind_x_e_N)
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
    
def solve_potential_periodic(n_x, n_N):
    
    #print("n_x: " + str(n_x))
    
    nvec = np.transpose(np.matrix([n_N]))
    
    #k = 2*np.pi/  -n_x/2
    k_range = np.linspace(-np.pi,np.pi*(1-2/n_x),n_x)
    
    #print("k_range: " + str(k_range))
    
    kvec = np.matrix(np.transpose([k_range]))
    j = np.matrix([np.arange(n_x)])

    #print(np.shape(nvec))
    #print(np.shape(kvec))
    #print(np.shape(j))
    step1 = np.exp(-1j*kvec*j)*nvec # Fourier transform of density as function of k

    step2_phi = np.matrix(np.diag(1/k_range**2))*np.matrix(np.exp(1j*kvec*j)) 
    step2_E = np.matrix(np.diag(1/(1j*k_range)))*np.matrix(np.exp(1j*kvec*j))
    
    step3_phi = np.transpose(step1)*step2_phi
    step3_E = np.transpose(step1)*step2_E
    
    phi_N = np.array(1/n_x*step3_phi)
    efield_N = np.array(1/n_x*step3_E, dtype=float)
    
    #print(phi_N[0,:])
    print(efield_N[0,:])
    
    return(phi_N[0,:], efield_N[0,:])

    
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

def integrate_motion(n_x, delta_x, delta_t, E_N, x_i_N, x_e_N, v_i_N, v_e_N, q_i_N, q_e_N, m_i_N, m_e_N):
    
    # Interpolate E-field to each particle
    # np.interp uses first-order interpolator, needed when combined w/ first-order charge-to-grid
    E_interp_i_N = np.interp(x_i_N, np.arange(n_x), E_N)
    E_interp_e_N = np.interp(x_e_N, np.arange(n_x), E_N)
    
    # Calculate new velocity
    v_i_new_N = (q*q_i_N*delta_t)**2/(m_i_N*eps0*delta_x)*E_interp_i_N + v_i_N
#    print(E_interp_i_N)
#    print(v_i_new_N)
    v_e_new_N = -(q*q_e_N*delta_t)**2/(m_e_N*eps0*delta_x)*E_interp_e_N + v_e_N
    
    # Calculate new position with periodic boundaries
    x_i_new_N = (x_i_N + v_i_new_N) % (n_x-1)
    x_e_new_N = (x_e_N + v_e_new_N) % (n_x-1)
    
    # Check if particle out of bounds - if so, make its position/velocity NaN
#    i_outofbounds_L = x_i_new_N<=0
#    i_outofbounds_U = x_i_new_N>=n_x
#    i_outofbounds_both = np.any(np.array([i_outofbounds_L, i_outofbounds_U]), axis=0)
#    x_i_new_N[i_outofbounds_both] = np.nan
#    v_i_new_N[i_outofbounds_both] = np.nan
#
#    e_outofbounds_L = x_e_new_N<=0
#    e_outofbounds_U = x_e_new_N>=n_x
#    e_outofbounds_both = np.any(np.array([e_outofbounds_L, e_outofbounds_U]), axis=0)
#    x_e_new_N[e_outofbounds_both] = np.nan
#    v_e_new_N[e_outofbounds_both] = np.nan
    
    # Return new values
    return(x_i_new_N, x_e_new_N, v_i_new_N, v_e_new_N)
    
def delete_edged_particles(n_x, x_N, v_N, q_N, m_N):
    
    margin = 1.0 # Just to be safe - delete if within 1 cell of edge

    indices = np.concatenate((np.where(x_N <= margin)[0], np.where(x_N >= (n_x-1-margin))[0]))
    
    x_N_new = np.delete(x_N, indices)
    v_N_new = np.delete(v_N, indices)
    q_N_new = np.delete(q_N, indices)
    m_N_new = np.delete(m_N, indices)
    
    num_deletions = np.size(indices)
    
    if num_deletions > 0:
        print('Deleted ' + str(np.size(indices)) + ' particles at boundary')
    
    return(x_N_new, v_N_new, q_N_new, m_N_new)

def phase_plot(x_i_N, x_e_N, v_i_N, v_e_N, tstep, axes):

    plt.figure()
    plot_ions = plt.scatter(x_i_N, v_i_N)
    plot_electrons = plt.scatter(x_e_N, v_e_N)
    plt.xlim(axes[0:2])
    plt.ylim(axes[2:4])
    plt.savefig('img/sct_' + str(tstep) + '.png')
    plt.close()



# Tests...
if safe_to_run:

    x_i_N = np.linspace(chunk_start,chunk_end,NP)
    x_e_N = np.linspace(chunk_start+delta_x/2,chunk_end+delta_x/2,NP)
    #print(x_i_N)
    #print(x_e_N)
    
    # Charge arrays
    q_i_N = np.ones(np.size(x_i_N))
    q_e_N = np.ones(np.size(x_e_N))
    m_i_N = m_i*np.ones(np.size(x_i_N))
    m_e_N = m_e*np.ones(np.size(x_e_N))
    
    v_i_N = quiet_start.maxwellian(NP, 0, vthi)
    #print(v_i_N)
    v_e_N = quiet_start.maxwellian(NP, vthe*10, vthe)
    #print(v_e_N)
    #v_i_N = np.zeros(np.size(x_i_N))
    #v_e_N = -0.1*np.ones(np.size(x_i_N))
    
    
    # Nondimensionalizing BCs...
    V_L_N = eps0/(q*delta_x) * V_L
    V_R_N = eps0/(q*delta_x) * V_R
    
    # Allocate time history of particle trajectories
    #x_i_N_T = np.zeros([n_tsteps, np.size(x_i_N)])
    #x_e_N_T = np.zeros([n_tsteps, np.size(x_i_N)])
    # Main Loop
        
    for t_N in np.arange(n_tsteps):
        print('evaluating timestep ' + str(t_N))
        n_N = interpolate_charges(n_x, x_i_N, x_e_N, q_i_N, q_e_N)
        
        plt.figure()
        plt.plot(np.arange(0,n_x,1), n_N)
        plt.savefig("img/rho_N" + str(t_N) + ".png")
        plt.close()
        
        phi_N, E_N = solve_potential_periodic(n_x, n_N)
        #phi_N = solve_potential(n_x, n_N, V_L_N, V_R_N)
        
        # E_N = solve_efield_from_potential(n_x, phi_N, V_L_N, V_R_N)
        x_i_N, x_e_N, v_i_N, v_e_N = integrate_motion(n_x, delta_x, delta_t, E_N, x_i_N, x_e_N, v_i_N, v_e_N, q_i_N, q_e_N, m_i_N, m_e_N)
        
        # Save particle positions
        #x_i_N_T[t_N,:] = x_i_N
        #x_e_N_T[t_N,:] = x_e_N
        
        # Delete edge electrons (leaving ions for now)
        # x_e_N, v_e_N, q_e_N, m_e_N = delete_edged_particles(n_x, x_e_N, v_e_N, q_e_N, m_e_N)
        
        # Plot particle positions
        phase_plot(x_i_N, x_e_N, v_i_N, v_e_N, t_N, [0, 1000, -5, 5])
    
    #t = np.arange(n_tsteps)*delta_t




#plt.figure()
#plt.plot(t,x_i_N_T)
#plt.figure()
#plt.plot(t,x_e_N_T)