import configparser

filename = input("Specify config base filename: ")

config = configparser.ConfigParser()

# case settings
config['CASE'] = {}
config['CASE']['profile_name'] = input("Specify case profile name: ")

# Default values changeable in file
config['FLAGS'] = {}
config['FLAGS']['verbose'] = "0"
config['FLAGS']['ic_override'] = "0"
config['FLAGS']['vel_override'] = "0"
config['FLAGS']['plot_all'] = "0"

# Plot flags
config['PLOTS'] = {}
config['PLOTS']['efield'] = "0"
config['PLOTS']['phi'] = "0"
config['PLOTS']['density'] = "0"
config['PLOTS']['energy'] = "0"
config['PLOTS']['all_debug'] = "0"
config['PLOTS']['phase'] = "1"
config['PLOTS']['fft'] = "0"

# Save data flags
config['SAVE'] = {}
config['SAVE']['efield'] = "1"
config['SAVE']['phi'] = "1"
config['SAVE']['energy'] = "1"
config['SAVE']['pos'] = "1"
config['SAVE']['vel'] = "1"
config['SAVE']['save_every_ntsteps'] = input("Specify number of timesteps per data-save: ")

# Values to prompt user for
config["SETTINGS"] = {}
config['SETTINGS']['n0'] = input("Specify background density n0: ")
config['SETTINGS']['mass_ratio'] = input("Specify ion-to-electron mass ratio (physical is 1836.15): ")
config['SETTINGS']['M'] = input("Specify number of particles per macroparticle: ")
config['SETTINGS']['Ti'] = input("Specify ion temperature Ti: ")
config['SETTINGS']['Te'] = input("Specify electron temperature Te: ")
config['SETTINGS']['ld_per_dx'] = input("Specify number of lambda_ds per delta_x: ")
config['SETTINGS']['dt_per_wp'] = input("Specify number of delta_ts per omega_p: ")
config['SETTINGS']['n_x'] = input("Specify number of delta_xs in domain length: ")
config['SETTINGS']['n_tsteps'] = input("Specify number of timesteps: ")
with open(filename + '.txt', 'w') as configfile:
    config.write(configfile)
