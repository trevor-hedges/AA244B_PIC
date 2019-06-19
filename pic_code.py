import numpy as np
import numpy.linalg as linalg

q_phys = 1.60218E-19 # C
eps0 = 8.8542E-12 # F/m

def interpolate_charges(n_x, x_i_N, x_e_N, mnum_i_N, mnum_e_N, lambda_d, delta_x_N, n0_N):
    """Given charge locations, solves for charge density on 1D grid
    Using first-order interpolation
    """

    x_i_NN = x_i_N/delta_x_N
    x_e_NN = x_e_N/delta_x_N

    ind_x_i_N = np.array(np.floor(x_i_NN), dtype=int)
    ind_x_e_N = np.array(np.floor(x_e_NN), dtype=int)

    chg_j_i_N = mnum_i_N*((ind_x_i_N+1)-x_i_NN)
    chg_jp1_i_N = mnum_i_N*(x_i_NN-ind_x_i_N)
    chg_j_e_N = -mnum_e_N*((ind_x_e_N+1)-x_e_NN)
    chg_jp1_e_N = -mnum_e_N*(x_e_NN-ind_x_e_N)

    # Count charges and assign to indices
    chg_N = np.zeros(n_x)
    np.add.at(chg_N, ind_x_i_N, chg_j_i_N)
    np.add.at(chg_N, (ind_x_i_N+1) % n_x, chg_jp1_i_N)
    np.add.at(chg_N, ind_x_e_N, chg_j_e_N)
    np.add.at(chg_N, (ind_x_e_N+1) % n_x, chg_jp1_e_N)

    # Scale as necessary
    rho_N = chg_N*lambda_d**2/(delta_x_N*n0_N)

    return(rho_N)

def solve_potential_periodic(n_x, rho_N, G_N, delta_x_N):
    """Given charge densities, solves for potential on 1D periodic grid"""

    # Construct A matrix
    A = np.diag(np.ones(n_x-2),1)+np.diag(-2*np.ones(n_x-1),0)+np.diag(np.ones(n_x-2),-1)
    A = -A
    b = rho_N[1:]

    # Scale
    b = b*delta_x_N**2

    b[0] += G_N
    b[-1] += G_N

    # Solve for phi
    phi_N_no0 = linalg.solve(A, b)

    # Add boundary point
    phi_N = np.zeros(n_x)
    phi_N[1:] += phi_N_no0
    phi_N[0] = G_N

    return(phi_N)

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

    # Calculate new positions at time t+delta_t with modulos to handle periodic boundaries
    x_i_new_N = (x_i_N + v_i_new_N*delta_t_N) % (n_x*delta_x_N)
    x_e_new_N = (x_e_N + v_e_new_N*delta_t_N) % (n_x*delta_x_N)

    # Return new values
    return(x_i_new_N, x_e_new_N, v_i_new_N, v_e_new_N)

def calculate_energy(v_i_old_N, v_e_old_N, v_i_new_N,
                     v_e_new_N, mnum_i_N, mnum_e_N, rho_N, phi_N, delta_x,
                     omega_p_phys, lambda_d_phys, n0, m_e_phys, mi_me_ratio,
                     x_i_old_N, x_e_old_N, delta_x_N, n_x, E_N, verbose=0):
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

    # Calculate dimensional electric potential energy
    PE = delta_x*omega_p_phys**2*lambda_d_phys**2*m_e_phys*n0/2*np.inner(rho_N, phi_N)

    # Calculate dimensional electric potential energy a different way
    x_i_old_NN = x_i_old_N/delta_x_N
    x_e_old_NN = x_e_old_N/delta_x_N

    # interpolate phi to particle positions
    phi_i_interp = np.interp(x_i_old_NN, np.arange(n_x), phi_N, period=n_x)
    phi_e_interp = np.interp(x_e_old_NN, np.arange(n_x), phi_N, period=n_x)

    PE2_i = 1/2*omega_p_phys**2*lambda_d_phys**2*m_e_phys*np.inner(mnum_i_N, phi_i_interp)
    PE2_e = -1/2*omega_p_phys**2*lambda_d_phys**2*m_e_phys*np.inner(mnum_e_N, phi_e_interp)
    PE2 = PE2_i + PE2_e

    # Calculate dimensional electric potential energy a third way
    E_dim = E_N*m_e_phys*lambda_d_phys*omega_p_phys**2/q_phys
    PE3 = 1/2*eps0*delta_x*np.inner(E_dim, E_dim)

    # Calculate total energy
    E_tot = KE_i + KE_e + PE
    E_tot2 = KE_i + KE_e + PE2

    # Normalize it
    En_N = E_tot/(m_e_phys*omega_p_phys**2)

    # Print stuff
    if verbose:
        print("Ion kinetic: " + str(KE_i))
        print("Electron kinetic: " + str(KE_e))
        print("Electric potential: " + str(PE))
        print("Electric potential2: " + str(PE2))
        print("Electric potential3: " + str(PE3))
        print("Total2: " + str(E_tot2))
        print("Total1_N: " + str(En_N))

    return(E_tot, KE_i, KE_e, PE)
