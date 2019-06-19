#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 08:27:32 2019

@author: Trevor Hedges
"""

import time
import os, sys
import configparser
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import dirs
plt.ioff()
import quiet_start
import pic_code
import pic_plot
import helper_functions

# Load config file specified
config = configparser.ConfigParser()
config.read("config.txt")

# Directory to save output files to
output_dir = 'output/' + config["CASE"]["profile_name"] + '/'

output_dirs = dirs.get_dirs(output_dir)

# helper_functions.check_make([output_dir + '/pos', output_dir + '/img/efield', output_dir + '/img/phi',
#                             output_dir + '/img/density', output_dir + '/img/energy',
#                             output_dir + '/img/fft', output_dir + '/img/all_debug',
#                             output_dir + '/img/phase'])

# Define flags
verbose = bool(int(config["FLAGS"]["verbose"]))
ic_override = bool(int(config['FLAGS']['ic_override'])) # Whether to ignore specifying plasma properties and load pre-set initial particle conditions
velocity_override = bool(int(config['FLAGS']['vel_override']))
plot_all = bool(int(config['FLAGS']['plot_all']))

# Define physical constants
q_phys = 1.60218E-19 # C
m_i_phys = 1.6726219E-27 # kg Assuming just protons
m_e_phys = 9.10938356E-31 # kg

eps0 = 8.8542E-12 # F/m
K = 1.3807E-23 # J/K
G = 0 # "Ground" potential

# "True" plasma density to simulate
n0 = float(config["SETTINGS"]["n0"])

q_to_m_i = q_phys/m_i_phys
q_to_m_e = q_phys/m_e_phys

# Define simulation constants
mi_me_ratio = float(config["SETTINGS"]["mass_ratio"])
m_e_sim = m_e_phys
m_i_sim = mi_me_ratio*m_e_phys # Mass of simulated ion

# Define number of particles per macroparticle
macroparticle_num = float(config["SETTINGS"]["M"])

print('simulated macroparticle me=' + str(macroparticle_num*m_e_sim) + " kg")
print('simulated macroparticle mi=' + str(macroparticle_num*m_i_sim) + " kg")

# Calculate electron Debye length
Ti = float(config["SETTINGS"]["Ti"]); # K
Te = float(config["SETTINGS"]["Te"]); # K
lambda_d_phys = np.sqrt(eps0*K*Te/(n0*q_phys**2)) # meters
print("lambda_d=" + str(lambda_d_phys) + " m")

# Calculate electron plasma frequency
omega_p_phys = np.sqrt(n0*q_phys**2/(eps0*m_e_phys))
print("omega_p=" + str(omega_p_phys) + " rad/s")

# Check # particles in debye sphere
N_d = n0*4/3*np.pi*lambda_d_phys**3
print("N_d=" + str(N_d))

# Define sim params (make these inputs later)
delta_x = lambda_d_phys/float(config["SETTINGS"]["ld_per_dx"]) # meters
delta_t = 1/(float(config["SETTINGS"]["dt_per_wp"])*omega_p_phys) # s^-1
print('delta_x=' + str(delta_x) + ' m')
print('delta_t=' + str(delta_t) + ' s')

n0_sim = n0/macroparticle_num # m^-1

# Nondimensional plasma density (per debye cube)
n0_N = n0*lambda_d_phys**3

lambda_d_sim = np.sqrt(eps0*K*Te*macroparticle_num/(n0_sim*(macroparticle_num*q_phys)**2))
print("lambda_d_sim = " + str(lambda_d_sim))
omega_p_sim = np.sqrt(n0_sim*(macroparticle_num*q_phys)**2/(eps0*m_e_sim*macroparticle_num))
print("omega_p_sim = " + str(omega_p_sim))

# Added 2...
vthi_sim_N = np.sqrt(2*K*Ti/m_i_sim)/(omega_p_phys*lambda_d_phys) # Normalized
vthe_sim_N = np.sqrt(2*K*Te/m_e_sim)/(omega_p_phys*lambda_d_phys)

print("Normalized ion thermal velocity = " + str(vthi_sim_N))
print("Normalized electron thermal velocity = " + str(vthe_sim_N))

n_x = int(config["SETTINGS"]["n_x"])
n_tsteps = int(config["SETTINGS"]["n_tsteps"]) # Power of 2 to make FFT at end go faster
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


# Tests...
if safe_to_run:

    # Nondimensionalize constants
    delta_x_N = delta_x/lambda_d_phys
    delta_t_N = delta_t*omega_p_phys

    if ic_override:

        #delta_x = 0.0053
        #delta_t = 1.7725E-7

        # Nondimensionalize constants
        #lambda_d_phys = 0.0526
        #omega_p_phys = 5.6416E5
        #delta_x_N = delta_x/lambda_d_phys
        #delta_t_N = delta_t*omega_p_phys


        #x_i_N = np.array([0.0714, 0.1429, 0.2143, 0.2857, 0.3571, 0.4286, 0.5, 0.5714, 0.6429, 0.7143, 0.7857, 0.8571, 0.9286, 0.99])
        #x_e_N = np.array([0.0357, 0.1071, 0.1786, 0.25, 0.3214, 0.3929, 0.4643, 0.5357, 0.6071, 0.6786, 0.7500, 0.8214, 0.8929, 0.9643])

        x_i_NN = np.array([n_x/10, n_x*6/10])
        x_e_NN = np.array([(n_x+1)/10, (n_x+1)*6/10])


        NP = np.size(x_i_NN)

        #x_i_NN = np.array([4.5, 4.9, 5.0, 5.1, 5.5, 6, 6.5, 8])
        #x_e_NN = np.array([1, 1.4, 1.6, 2.6, 3.5, 4.7, 7, 9])
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
            jumble_factor = 0.1
            x_i_NN = (np.linspace(margin,n_x-margin,NP) + jumble_factor*np.random.normal(size=NP)) % n_x
            x_e_NN = (np.linspace(margin,n_x-margin,NP) + jumble_factor*np.random.normal(size=NP)) % n_x
        else:
            x_i_NN = np.linspace(margin,n_x-margin,NP) % n_x
            x_e_NN = np.linspace(n_x/(2*NP)+margin,n_x-margin,NP) % n_x
        x_i_N = x_i_NN*delta_x_N
        x_e_N = x_e_NN*delta_x_N

        if verbose:
            print("Initial ion locations: " + str(x_i_N))
            print("Initial electron locations: " + str(x_e_N))

        # Initialize particle velocities using not-quiet start technique
        t0 = time.time()
        v_i_N = quiet_start.maxwellian(NP, 0, vthi_sim_N)
        v_e_N = quiet_start.maxwellian(NP, 0, vthe_sim_N)
        t1 = time.time()
        if verbose:
            # Make this go to log file
            print("Initial velocity profiles: " + str(t1-t0) + " sec")

#        if verbose:
#            print("Initial ion normalized velocities: " + str(v_i_N))
#            print("Initial electron normalized velocities: " + str(v_e_N))
        if velocity_override:
            v_i_N = np.zeros(np.size(x_i_N))
            v_e_N = np.zeros(np.size(x_e_N))
            v_e_N[int(4*np.size(x_e_N)/10):int(6*np.size(x_e_N)/10)] = 0.1*delta_x_N/delta_t_N

    # Allocate total energy array
    Etot_vs_t = np.zeros(n_tsteps)
    KEi_vs_t = np.zeros(n_tsteps)
    KEe_vs_t = np.zeros(n_tsteps)
    PE_vs_t = np.zeros(n_tsteps)

    # Macroparticle number arrays
    mnum_i_N = macroparticle_num*np.ones(np.size(x_i_N))
    mnum_e_N = macroparticle_num*np.ones(np.size(x_e_N))

    # Nondimensionalize boundary conditions
    G_N = q_phys/(omega_p_phys**2*m_e_phys*lambda_d_phys**2) * G

    # Allocate time history of particle trajectories
    x_i_N_T = np.zeros([n_tsteps, np.size(x_i_N)])
    x_e_N_T = np.zeros([n_tsteps, np.size(x_e_N)])
    v_i_N_T = np.zeros([n_tsteps, np.size(v_i_N)])
    v_e_N_T = np.zeros([n_tsteps, np.size(v_e_N)])
    E_N_T = np.zeros([n_tsteps, n_x])
    phi_N_T = np.zeros([n_tsteps, n_x])

    # Main Loop
    for t_N in np.arange(n_tsteps):
        if verbose:
            print('evaluating timestep ' + str(t_N))

        t0 = time.time()
        rho_N = pic_code.interpolate_charges(n_x, x_i_N, x_e_N, mnum_i_N, mnum_e_N, lambda_d_phys, delta_x_N, n0_N)
        t1 = time.time()
        if verbose:
            print("Interpolated charges: " + str(t1-t0) + " sec")

        t0 = time.time()
        phi_N = pic_code.solve_potential_periodic(n_x, rho_N, G_N, delta_x_N)
        t1 = time.time()
        if verbose:
            print("Solved for potential function: " + str(t1-t0) + " sec")

        t0 = time.time()
        E_N = pic_code.solve_efield_from_potential_periodic(phi_N, delta_x_N)
        t1 = time.time()
        if verbose:
            print("Solved for E-field: " + str(t1-t0) + " sec")

        t0 = time.time()
        x_i_new_N, x_e_new_N, v_i_new_N, v_e_new_N = pic_code.integrate_motion(n_x, E_N, x_i_N, x_e_N, v_i_N, v_e_N, delta_x_N, delta_t_N, mi_me_ratio)
        t1 = time.time()
        if verbose:
            print("Integrated motion: " + str(t1-t0) + " sec")

        # Calculate total energy
        t0 = time.time()
        Etot_vs_t[t_N], KEi_vs_t[t_N], KEe_vs_t[t_N], PE_vs_t[t_N] = pic_code.calculate_energy(v_i_N, v_e_N, v_i_new_N,
                     v_e_new_N, mnum_i_N, mnum_e_N, rho_N, phi_N, delta_x,
                     omega_p_phys, lambda_d_phys, n0, m_e_phys, mi_me_ratio, x_i_N,
                     x_e_N, delta_x_N, n_x, E_N, verbose)
        t1 = time.time()
        if verbose:
            print("Calculated total energy: " + str(t1-t0) + " sec")

        # Print to console
        if verbose:
            print("Total system energy = " + str(Etot_vs_t[t_N]) + " J/m^2")

        # Save particle positions
        t0 = time.time()
        x_i_N_T[t_N,:] = x_i_N
        x_e_N_T[t_N,:] = x_e_N
        v_i_N_T[t_N,:] = v_i_N
        v_e_N_T[t_N,:] = v_e_N
        E_N_T[t_N,:] = E_N
        phi_N_T[t_N,:] = phi_N
        t1 = time.time()
        if verbose:
            print("Saved particle positions: " + str(t1-t0) + " sec")

        if plot_all:
            t0 = time.time()
            # Plot normalized E-field
            plt.figure()
            plt.plot(np.arange(0,n_x,1), E_N)
            plt.savefig(output_dirs['e_field'] + "e_N" + str(t_N) + ".png")
            plt.close()

            plt.figure()
            plt.plot(np.arange(0,n_x,1), phi_N)
            plt.savefig(output_dirs['phi'] + "phi_N" + str(t_N) + ".png")
            plt.close()

            # Plot normalized charge density
            plt.figure()
            plt.plot(np.arange(0,n_x,1), rho_N)
            plt.savefig(output_dirs['density'] + "rho_N" + str(t_N) + ".png")
            plt.close()
            t1 = time.time()
            if verbose:
                print("Saved E-field, phi, rho plots: " + str(t1-t0) + " sec")

        # Plot particle positions
        if plot_all:
            t0 = time.time()
            x_i_NN = x_i_N/delta_x_N
            x_e_NN = x_e_N/delta_x_N
            v_i_NN = v_i_N*delta_t_N/delta_x_N
            v_e_NN = v_e_N*delta_t_N/delta_x_N
            pic_plot.phase_plot(x_i_NN, x_e_NN, v_i_NN, v_e_NN, t_N, [0, n_x, -1, 1], output_dirs['phase'])
            t1 = time.time()
            if verbose:
                print("Saved phase plot: " + str(t1-t0) + " sec")

            # Make plots of particles + fields superimposed at time t=0
            t0 = time.time()
            pic_plot.all_plot(NP, n_x, x_i_NN, x_e_NN, E_N, phi_N, t_N, [0, n_x, -50000, 50000], output_dirs['all_debug'])
            t1 = time.time()
            if verbose:
                print("Saved all-plot: " + str(t1-t0) + " sec")

        # Set new
        t0 = time.time()
        x_i_N = x_i_new_N
        x_e_N = x_e_new_N
        v_i_N = v_i_new_N
        v_e_N = v_e_new_N
        t1 = time.time()
        if verbose:
            print("Updated positions and velocities: " + str(t1-t0) + " sec")

    # Save plot of total energy
    pic_plot.energy_plot(n_tsteps, Etot_vs_t, KEi_vs_t, KEe_vs_t, PE_vs_t, output_dirs['energy'])

    np.save(output_dirs['data'] + "ion_positions.npy", x_i_N_T)
    np.save(output_dirs['data'] + "electron_positions.npy", x_e_N_T)
    np.save(output_dirs['data'] + "ion_velocities.npy", v_i_N_T)
    np.save(output_dirs['data'] + "electron_velocities.npy", v_e_N_T)
    np.save(output_dirs['data'] + "Efield.npy", E_N_T)
    np.save(output_dirs['data'] + "phifield.npy", phi_N_T)
    np.save(output_dirs['data'] + "total_energy.npy", Etot_vs_t)
    np.save(output_dirs['data'] + "KEi.npy", KEi_vs_t)
    np.save(output_dirs['data'] + "KEe.npy", KEe_vs_t)
    np.save(output_dirs['data'] + "PE.npy", PE_vs_t)


    # FFT output position
    if NP < 1000:
        for p_n in np.arange(NP):
            pic_plot.fft_time_plot(n_tsteps, np.abs(np.fft.fft(x_e_N_T[:,p_n])), p_n, output_dirs['fft'])
