#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 08:27:32 2019

@author: Trevor Hedges
"""

import numpy as np
import matplotlib.pyplot as plt

import quiet_start

# Define flags
verbose = 1
bc_type = 1 # 0 = fixed potential, 1 = periodic, 2 = periodic + fourier
ic_override = 1 # Whether to ignore specifying plasma properties and load pre-set initial particle conditions

# Define physical constants
q_phys = 1.60218E-19 # C
m_i_phys = 1.6726219E-27 # kg Assuming just protons
m_e_phys = 9.10938356E-31 # kg

eps0 = 8.8542E-12 # F/m
K = 1.3807E-23 # J/K

# Define boundary conditions if necessary
if bc_type == 0:
    V_L = 0
    V_R = 0
elif bc_type == 1:
    G = 0 # "Ground" potential

# "True" plasma density to simulate
n0 = 5E6

q_to_m_i = q_phys/m_i_phys
q_to_m_e = q_phys/m_e_phys

# Define simulation constants
mi_me_ratio = 1
m_e_sim = m_e_phys
m_i_sim = mi_me_ratio*m_e_phys # Mass of simulated ion

# Define number of particles per macroparticle
macroparticle_num = 50000

print('simulated macroparticle me=' + str(macroparticle_num*m_e_sim) + " kg")
print('simulated macroparticle mi=' + str(macroparticle_num*m_i_sim) + " kg")

# Calculate electron Debye length
Ti = 50;
Te = 50;
lambda_d_phys = np.sqrt(eps0*K*Te/(n0*q_phys**2)) # meters
print("lambda_d=" + str(lambda_d_phys) + " m")

# Calculate electron plasma frequency
omega_p_phys = np.sqrt(n0*q_phys**2/(eps0*m_e_phys))
print("omega_p=" + str(omega_p_phys) + " rad/s")

# Check # particles in debye sphere
N_d = n0*4/3*np.pi*lambda_d_phys**3
print("N_d=" + str(N_d))

# Define sim params (make these inputs later)
delta_x = lambda_d_phys/20 # meters
delta_t = 0.001/omega_p_phys # s^-1
print('delta_x=' + str(delta_x) + ' m')
print('delta_t=' + str(delta_t) + ' s')

n0_sim = n0/macroparticle_num # m^-1

# Nondimensional plasma density (per debye cube)
n0_N = n0*lambda_d_phys**3

lambda_d_sim = np.sqrt(eps0*K*Te*macroparticle_num/(n0_sim*(macroparticle_num*q_phys)**2))
print("lambda_d_sim = " + str(lambda_d_sim))
omega_p_sim = np.sqrt(n0_sim*(macroparticle_num*q_phys)**2/(eps0*m_e_sim*macroparticle_num))
print("omega_p_sim = " + str(omega_p_sim))

vthi_sim = np.sqrt(K*Ti/m_i_sim)*delta_t/delta_x # Normalized
vthe_sim = np.sqrt(K*Te/m_e_sim)*delta_t/delta_x

print("Normalized ion thermal velocity = " + str(vthi_sim))
print("Normalized electron thermal velocity = " + str(vthe_sim))

n_x = 11
n_tsteps = 1000
t_tot = delta_t*n_tsteps
plasma_cycles = (omega_p_phys/(2*np.pi))*t_tot
print("Total time to be simulated: " + str(t_tot) + " seconds; " + str(plasma_cycles) + " plasma oscillations")

# Test case initial conditions...
L = delta_x*n_x
NP = int(np.round(n0_sim*L))

print('number of each species=' + str(NP))
print("L="+str(L)+" m")

# Check plasma criteria
if lambda_d_phys > L:
    print("WARNING: Plasma criteria lambda_d << L is DEFINITELY not met.")
elif lambda_d_phys > L/10:
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

if NP <= 1000000:
    safe_to_run = True
else:
    safe_to_run = False
    print('WAY TOO MANY PARTICLES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')



def interpolate_charges(n_x, x_i_N, x_e_N, mnum_i_N, mnum_e_N, lambda_d, delta_x_N, n0_N):
    """Given charge locations, solves for charge density on 1D grid
    Using first-order interpolation
    """
    
    x_i_NN = x_i_N/delta_x_N
    x_e_NN = x_e_N/delta_x_N
    
    ind_x_i_N = np.array(np.floor(x_i_NN), dtype=int)
    ind_x_e_N = np.array(np.floor(x_e_NN), dtype=int)

    #ind_x_i_NP1 = (ind_x_i_N+1) % n_x
    #ind_x_e_NP1 = (ind_x_e_N+1) % n_x
    
    # print('ind_x_i: ' + str(ind_x_i_N))
    # print('ind_x_e: ' + str(ind_x_e_N))
#    
    chg_j_i_N = mnum_i_N*((ind_x_i_N+1)-x_i_NN)
    chg_jp1_i_N = mnum_i_N*(x_i_NN-ind_x_i_N)
    chg_j_e_N = -mnum_e_N*((ind_x_e_N+1)-x_e_NN)
    chg_jp1_e_N = -mnum_e_N*(x_e_NN-ind_x_e_N)

#    chg_j_i_N = 1*((ind_x_i_N+1)-x_i_N)
#    chg_jp1_i_N = 1*(x_i_N-ind_x_i_N)
#    chg_j_e_N = -1*((ind_x_e_N+1)-x_e_N)
#    chg_jp1_e_N = -1*(x_e_N-ind_x_e_N)


    # Count charges and assign to indices
    chg_N = np.zeros(n_x)
    np.add.at(chg_N, ind_x_i_N, chg_j_i_N)
    np.add.at(chg_N, (ind_x_i_N+1) % n_x, chg_jp1_i_N)
    np.add.at(chg_N, ind_x_e_N, chg_j_e_N)
    np.add.at(chg_N, (ind_x_e_N+1) % n_x, chg_jp1_e_N)
    
    # Scale as necessary
    rho_N = chg_N*lambda_d**2/(delta_x_N*n0_N)
    
    return(rho_N)
    
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
    
def solve_potential_periodic(n_x, rho_N, G_N, delta_x_N):
    """Given charge densities, solves for potential on 1D periodic grid"""
    
    # Construct A matrix
    A = np.diag(np.ones(n_x-2),1)+np.diag(-2*np.ones(n_x-1),0)+np.diag(np.ones(n_x-2),-1)
#    A[-1, 0] = 1
#    A[0, -1] = 1
    A = -A
    b = rho_N[1:]
    
    # Scale
    b = b*delta_x_N**2
    
    b[0] += G_N
    b[-1] += G_N
    
    # Solve for phi
    phi_N_no0 = np.linalg.solve(A, b)
    
    # Add boundary point
    phi_N = np.zeros(n_x)
    phi_N[1:] += phi_N_no0
    phi_N[0] = G_N
    
    return(phi_N)
    
def solve_potential_fourier(n_x, n_N):
    
    #print("n_x: " + str(n_x))
    
    nvec = np.transpose(np.matrix([n_N]))
    
    k_range = np.linspace(-np.pi,np.pi*(1-2/n_x),n_x)
    
    kvec = np.matrix(np.transpose([k_range]))
    j = np.matrix([np.arange(n_x)])

    step1 = np.exp(-1j*kvec*j)*nvec # Fourier transform of density as function of k

    step2_phi = np.matrix(np.diag(1/k_range**2))*np.matrix(np.exp(1j*kvec*j)) 
    step2_E = np.matrix(np.diag(1/(1j*k_range)))*np.matrix(np.exp(1j*kvec*j))
    
    step3_phi = np.transpose(step1)*step2_phi
    step3_E = np.transpose(step1)*step2_E
    
    phi_N = np.array(1/n_x*step3_phi, dtype=float)
    efield_N = np.array(1/n_x*step3_E, dtype=float)
        
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

def solve_efield_from_potential_periodic(phi_N, delta_x_N):
    """Given potential field, solves for e-field in 1D"""
    
    # Perform calculation
    E_N = -(np.roll(phi_N, -1) - np.roll(phi_N, 1))/(2*delta_x_N)
    
    # Return E-field    
    return(E_N)


def integrate_motion(n_x, E_N, x_i_N, x_e_N, v_i_N, v_e_N, delta_x_N, delta_t_N, mi_me_ratio_N):
    """Use Leapfrog method to integrate new particle velocities and positions """
    
    # Interpolate E-field to each particle
    # np.interp uses first-order interpolator, needed when combined w/ first-order charge-to-grid
    x_i_NN = x_i_N/delta_x_N
    x_e_NN = x_e_N/delta_x_N
    
    E_interp_i_N = np.interp(x_i_NN, np.arange(n_x), E_N, period=n_x)
    E_interp_e_N = np.interp(x_e_NN, np.arange(n_x), E_N, period=n_x)
    
    # Calculate new velocities at time t+delta_t/2
    v_i_new_N = 1/mi_me_ratio_N*E_interp_i_N*delta_t_N+v_i_N
    v_e_new_N = -E_interp_e_N*delta_t_N+v_e_N

#    v_i_new_N = (q_phys*delta_t)**2/(m_i_sim*eps0*delta_x)*E_interp_i_N + v_i_N
#    v_e_new_N = -(q_phys*delta_t)**2/(m_e_sim*eps0*delta_x)*E_interp_e_N + v_e_N
    
    # Calculate new positions at time t+delta_t with modulos to handle periodic boundaries
    x_i_new_N = (x_i_N + v_i_new_N*delta_t_N) % (n_x*delta_x_N)
    x_e_new_N = (x_e_N + v_e_new_N*delta_t_N) % (n_x*delta_x_N)
        
    # Return new values
    return(x_i_new_N, x_e_new_N, v_i_new_N, v_e_new_N)
        
def calculate_energy(v_i_old_N, v_e_old_N, v_i_new_N,
                     v_e_new_N, mnum_i_N, mnum_e_N, rho_N, phi_N, delta_x,
                     omega_p_phys, lambda_d_phys, n0, m_e_phys, mi_me_ratio):
    """Calculate energy in system at time t. INPUTS: x_i(t), x_e(t),
    v_i(t-delta_t/2), v_e(t-delta_t/2), v_i(t+delta_t/2), v_e(t+delta_t/2)
    macroparticle number (ions), macroparticle number (electrons),
    charge density, electric potential """
    
    # Calculate velocity squared at time t (approximate to O(delta_t^2))
    v_i_t_squared_N = v_i_old_N*v_i_new_N
    v_e_t_squared_N = v_e_old_N*v_e_new_N
    
    # Calculate dimensional kinetic energy for each species    
    KE_i = 1/2*m_e_phys*mi_me_ratio*lambda_d_phys**2*omega_p_phys**2*np.inner(mnum_i_N, v_i_t_squared_N)
    KE_e = 1/2*m_e_phys*lambda_d_phys**2*omega_p_phys**2*np.inner(mnum_e_N, v_e_t_squared_N)
    
    #np.inner(mnum_i_N, v_i_t_squared)
    #KE_e_N = 1/2*np.inner(mnum_e_N, v_e_t_squared)
    
    # Calculate dimensional electric potential energy
    PE = delta_x*omega_p_phys**2*lambda_d_phys**2*m_e_phys*n0/2*np.inner(rho_N, phi_N)
    
    #PE_N = 1/2*q_phys**2/eps0*np.inner(chg_N, phi_N)
    
    # Calculate total energy 
    E_tot = KE_i + KE_e + PE
    
    # Print stuff
    print("Ion kinetic: " + str(KE_i))
    print("Electron kinetic: " + str(KE_e))
    print("Electric potential: " + str(PE))
    
    return(E_tot, KE_i, KE_e, PE)

def phase_plot(x_i_N, x_e_N, v_i_N, v_e_N, tstep, axes):

    plt.figure()
    plot_ions = plt.scatter(x_i_N, v_i_N)
    plot_electrons = plt.scatter(x_e_N, v_e_N)
    plt.xlim(axes[0:2])
    plt.ylim(axes[2:4])
    plt.savefig('img/phase/sct_' + str(tstep) + '.png')
    plt.close()

def all_plot(NP, n_x, x_i_N, x_e_N, E_N, phi_N, tstep, axes):
    
    xr = np.linspace(0,n_x-1,n_x)
    xrp1 = np.linspace(0,n_x,n_x+1)
    
    plt.figure()
    plt.scatter(x_i_N, np.zeros(NP), color='red', label='ions')
    plt.scatter(x_e_N, np.zeros(NP), color='blue', label='electrons')
    plt.plot(xrp1, np.concatenate((phi_N/10, np.array([phi_N[0]/10]))), color='red', label='phi')
    plt.plot(xrp1, np.concatenate((E_N, np.array([E_N[0]]))), color='blue', label='E')
    plt.xlim(axes[0:2])
#    plt.ylim(axes[2:4])
    plt.legend()
    plt.savefig('img/all_debug/all_' + str(tstep) + '.png')
    plt.close()
    
def energy_plot(n_tsteps, Etot_vs_t, KEi_vs_t, KEe_vs_t, PE_vs_t):
    
    plt.figure()
    plt.plot(np.arange(n_tsteps), Etot_vs_t, label='Total energy')
    plt.plot(np.arange(n_tsteps), KEi_vs_t, label='Ion KE')
    plt.plot(np.arange(n_tsteps), KEe_vs_t, label='Electron KE')
    plt.plot(np.arange(n_tsteps), KEi_vs_t+KEe_vs_t, label='Total KE')
    plt.plot(np.arange(n_tsteps), PE_vs_t, label='Electric PE')
    plt.title("Total energy vs time")
    plt.xlabel("Timestep")
    plt.ylabel("Energy (J/m^2)")
    plt.legend()
    plt.savefig("img/energy/energy_vs_time.png")
    plt.close()

# Tests...
if safe_to_run:

    # Nondimensionalize constants
    delta_x_N = delta_x/lambda_d_phys
    delta_t_N = delta_t*omega_p_phys
    
    if ic_override:
        
        delta_x = 1
        delta_t = 3E-6
        
        # Nondimensionalize constants
        delta_x_N = delta_x/lambda_d_phys
        delta_t_N = delta_t*omega_p_phys
        
        NP = 8
        x_i_NN = np.array([4.5, 4.9, 5.0, 5.1, 5.5, 6, 6.5, 8])
        x_e_NN = np.array([1, 1.4, 1.6, 2.6, 3.5, 4.7, 7, 9])
        #v_i_NN = np.array([0,0,0,0,0,0,0,0])
        #v_e_NN = np.array([0,0,0,0,0,0,0,0])
        
        x_i_N = x_i_NN*delta_x_N
        x_e_N = x_e_NN*delta_x_N
        v_i_N = np.zeros(NP)
        v_e_N = np.zeros(NP)
        
    else:
        # Initialize particle positions TODO: add some randomization (?)
        margin = 0
    #    x_i_N = np.arange(margin,n_x-margin,n_x/NP)
    #    x_e_N = np.arange(n_x/(2*NP)+margin,n_x-margin,n_x/NP)
        jumble_pos = 1
        if jumble_pos:
            jumble_factor = 0.5
            x_i_N = (np.linspace(margin,n_x-margin,NP) + jumble_factor*np.random.normal(size=NP)) % n_x
            x_e_N = (np.linspace(margin,n_x-margin,NP) + jumble_factor*np.random.normal(size=NP)) % n_x
        else:
            x_i_N = np.linspace(margin,n_x-margin,NP)
            x_e_N = np.linspace(n_x/(2*NP)+margin,n_x-margin,NP)
        
        if verbose:
            print("Initial ion locations: " + str(x_i_N))
            print("Initial electron locations: " + str(x_e_N))
            
        # Initialize particle velocities using quiet start technique    
        v_i_N = quiet_start.maxwellian(NP, 0, vthe_sim)
        v_e_N = quiet_start.maxwellian(NP, vthe_sim*0, vthe_sim)    
        if verbose:
            print("Initial ion normalized velocities: " + str(v_i_N))
            print("Initial electron normalized velocities: " + str(v_e_N))

    # Allocate total energy array
    Etot_vs_t = np.zeros(n_tsteps)
    KEi_vs_t = np.zeros(n_tsteps)
    KEe_vs_t = np.zeros(n_tsteps)
    PE_vs_t = np.zeros(n_tsteps)

    # Macroparticle number arrays
    mnum_i_N = macroparticle_num*np.ones(np.size(x_i_N))
    mnum_e_N = macroparticle_num*np.ones(np.size(x_e_N))

    if bc_type == 0:
        # Nondimensionalize boundary conditions
        V_L_N = eps0/(q_phys*delta_x) * V_L
        V_R_N = eps0/(q_phys*delta_x) * V_R
    elif bc_type == 1:
        G_N = q_phys/(omega_p_phys**2*m_e_phys*lambda_d_phys**2) * G


    # Nondimensionalize constants
    #delta_x_N = delta_x/lambda_d_phys
    #delta_t_N = delta_t*omega_p_phys

    # Allocate time history of particle trajectories
    #x_i_N_T = np.zeros([n_tsteps, np.size(x_i_N)])
    #x_e_N_T = np.zeros([n_tsteps, np.size(x_i_N)])

    # Main Loop        
    for t_N in np.arange(n_tsteps):
        print('evaluating timestep ' + str(t_N))
        # (n_x, x_i_N, x_e_N, mnum_i_N, mnum_e_N, lambda_d, delta_x_N, n0_N):
        rho_N = interpolate_charges(n_x, x_i_N, x_e_N, mnum_i_N, mnum_e_N, lambda_d_phys, delta_x_N, n0_N)
        
        if bc_type==2:
            phi_N, E_N = solve_potential_fourier(n_x, rho_N)
        elif bc_type==1:
            phi_N = solve_potential_periodic(n_x, rho_N, G_N, delta_x_N)
            E_N = solve_efield_from_potential_periodic(phi_N, delta_x_N)
        else:
            phi_N = solve_potential(n_x, rho_N, V_L_N, V_R_N)
            E_N = solve_efield_from_potential(n_x, phi_N, V_L_N, V_R_N)
            
        x_i_new_N, x_e_new_N, v_i_new_N, v_e_new_N = integrate_motion(n_x, E_N, x_i_N, x_e_N, v_i_N, v_e_N, delta_x_N, delta_t_N, mi_me_ratio)
        
        # Calculate total energy
        #E_old = calculate_energy(x_i_N, x_e_N, v_i_N, v_e_N, v_i_new_N, v_e_new_N, mnum_i_N, mnum_e_N, chg_N, phi_N)
        Etot_vs_t[t_N], KEi_vs_t[t_N], KEe_vs_t[t_N], PE_vs_t[t_N] = calculate_energy(v_i_N, v_e_N, v_i_new_N,
                     v_e_new_N, mnum_i_N, mnum_e_N, rho_N, phi_N, delta_x,
                     omega_p_phys, lambda_d_phys, n0, m_e_phys, mi_me_ratio)
        
        # Print to console
        print("Total system energy = " + str(Etot_vs_t[t_N]) + " J/m^2")
        
        # Save particle positions
        #x_i_N_T[t_N,:] = x_i_N
        #x_e_N_T[t_N,:] = x_e_N
                
        # Plot normalized E-field
        plt.figure()
        plt.plot(np.arange(0,n_x,1), E_N)
        plt.savefig("img/efield/e_N" + str(t_N) + ".png")
        plt.close()
        
        plt.figure()
        plt.plot(np.arange(0,n_x,1), phi_N)
        plt.savefig("img/phi/phi_N" + str(t_N) + ".png")
        plt.close()
        
        # Plot normalized charge density
        plt.figure()
        plt.plot(np.arange(0,n_x,1), rho_N)
        plt.savefig("img/density/rho_N" + str(t_N) + ".png")
        plt.close()
        
        # Plot particle positions
        x_i_NN = x_i_N/delta_x_N
        x_e_NN = x_e_N/delta_x_N
        v_i_NN = v_i_N*delta_t_N/delta_x_N
        v_e_NN = v_e_N*delta_t_N/delta_x_N
        phase_plot(x_i_NN, x_e_NN, v_i_NN, v_e_NN, t_N, [0, n_x, -0.3, 0.3])
    
        # Make plots of particles + fields superimposed at time t=0
        all_plot(NP, n_x, x_i_NN, x_e_NN, E_N, phi_N, t_N, [0, n_x, -50000, 50000])
    
        # Set new
        x_i_N = x_i_new_N
        x_e_N = x_e_new_N
        v_i_N = v_i_new_N
        v_e_N = v_e_new_N
        
    #t = np.arange(n_tsteps)*delta_t
    # Save plot of total energy
    energy_plot(n_tsteps, Etot_vs_t, KEi_vs_t, KEe_vs_t, PE_vs_t)




#plt.figure()
#plt.plot(t,x_i_N_T)
#plt.figure()
#plt.plot(t,x_e_N_T)