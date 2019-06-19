import numpy as np
import matplotlib.pyplot as plt

def phase_plot(x_i_N, x_e_N, v_i_N, v_e_N, tstep, axes):

    plt.figure()
    plt.scatter(x_i_N, v_i_N)
    plt.scatter(x_e_N, v_e_N)
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

def fft_time_plot(n_tsteps, fft_abs, p_n):

    plt.figure()
    plt.plot(np.arange(n_tsteps), fft_abs)
    plt.savefig("img/fft/fft_e_" + str(p_n) + ".png")
    plt.close()

    np.save("img/fft/fft_e_" + str(p_n), fft_abs)
