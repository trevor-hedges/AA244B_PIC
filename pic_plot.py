import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def make_1D_vs_t_NN_movie(F_T, t_NN_T, var, output_dir, fps=10, dpi=100):

    # Get number of tsteps and deltaxs
    n_tsteps_saved = np.shape(F_T)[0]
    n_x = np.shape(F_T)[1]

    # Get min and max values for plotting
    F_T_min = np.min(F_T)
    F_T_max = np.max(F_T)
    axis_F_T = [0, n_x, F_T_min, F_T_max]

    # Iterate through all timesteps that were saved
    #f_movie_fig = plt.figure(figsize=(16,18))
    f_movie_fig = plt.figure()

    FFMpegWriter = animation.writers['ffmpeg_file']
    metadata = dict(title='', artist='Trevor Hedges', comment='')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    with writer.saving(f_movie_fig, output_dir + var + ".mp4", dpi):
        for t_entry in np.arange(n_tsteps_saved):

            f_movie_fig.clf()
            f_movie_plt = f_movie_fig.add_subplot(111)
            f_movie_plt.plot(np.arange(0,n_x+1,1), np.concatenate((F_T[t_entry,:], [F_T[t_entry,0]])))
            f_movie_plt.set_xlim(axis_F_T[0], axis_F_T[1])
            f_movie_plt.set_ylim(axis_F_T[2], axis_F_T[3])
            f_movie_plt.set_title(r'$\tilde{\tilde{t}}=$' + str(t_NN_T[t_entry]))
            f_movie_plt.set_xlabel(r'$\tilde{\tilde{x}}$')
            f_movie_plt.set_ylabel(var)
            f_movie_plt.grid()
            writer.grab_frame()


def phase_plot(x_i_N, x_e_N, v_i_N, v_e_N, tstep, axes, output_dir):

    plt.figure()
    plt.scatter(x_i_N, v_i_N)
    plt.scatter(x_e_N, v_e_N)
    plt.xlim(axes[0:2])
    plt.ylim(axes[2:4])
    plt.savefig(output_dir + 'sct_' + str(tstep) + '.png')
    plt.close()

def all_plot(NP, n_x, x_i_N, x_e_N, E_N, phi_N, tstep, axes, output_dir):

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
    plt.savefig(output_dir + str(tstep) + '.png')
    plt.close()

def energy_plot(t_N_T, Etot_vs_t, KEi_vs_t, KEe_vs_t, PE_vs_t, output_dir):

    plt.figure()
    plt.plot(t_N_T, Etot_vs_t, label='Total energy')
    plt.plot(t_N_T, KEi_vs_t, label='Ion KE')
    plt.plot(t_N_T, KEe_vs_t, label='Electron KE')
    plt.plot(t_N_T, KEi_vs_t+KEe_vs_t, label='Total KE')
    plt.plot(t_N_T, PE_vs_t, label='Electric PE')
    plt.title("Total energy vs time")
    plt.xlabel("Timestep")
    plt.ylabel("Energy (J/m^2)")
    plt.legend()
    plt.savefig(output_dir + "energy_vs_time.png")
    plt.close()

def fft_time_plot(n_tsteps, fft_abs, p_n, output_dir):

    plt.figure()
    plt.plot(np.arange(n_tsteps), fft_abs)
    plt.savefig(output_dir + "fft_e_" + str(p_n) + ".png")
    plt.close()

    np.save(output_dir + "fft_e_" + str(p_n), fft_abs)
