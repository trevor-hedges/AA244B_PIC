import os, os.path
import argparse
import configparser
import dirs

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import pic_plot

# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("config_file", help="Name of config file to run")
args = argparser.parse_args()
# Make sure config file has text file extension
if os.path.splitext(args.config_file)[1] != '.txt':
    config_file = args.config_file + ".txt"
else:
    config_file = args.config_file

# Load config file specified
config = configparser.ConfigParser()
config.read(config_file)

# Directory to save output files to
output_dir = 'output/' + config["CASE"]["profile_name"] + '/'
output_dirs = dirs.get_dirs(output_dir)

# Determine what to plot
plot_efield = bool(int(config["PLOTS"]["efield"]))
plot_phi = bool(int(config["PLOTS"]["phi"]))
plot_density = bool(int(config["PLOTS"]["density"]))
plot_energy = bool(int(config["PLOTS"]["energy"]))
plot_alldebug = bool(int(config["PLOTS"]["all_debug"]))
plot_phase = bool(int(config["PLOTS"]["phase"]))

# Load data files
t_NN_T = np.load(output_dirs['data'] + 'time.npy')

if plot_efield:
    E_N_T = np.load(output_dirs['data'] + 'e_field.npy')
    pic_plot.make_1D_vs_t_NN_movie(E_N_T, t_NN_T, "E", output_dirs['e_field'], fps=10)

if plot_phi:
    phi_N_T = np.load(output_dirs['data'] + 'phi_field.npy')
    pic_plot.make_1D_vs_t_NN_movie(phi_N_T, t_NN_T, "phi", output_dirs['phi'], fps=10)
