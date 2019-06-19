
def get_dirs(output_dir):

    pic_directories = {'output': output_dir,
                        'data': output_dir + 'pos/',
                        'plot': output_dir + 'img/',
                        'e_field': output_dir + 'img/efield/',
                        'phi': output_dir + 'img/phi/',
                        'rho': output_dir + 'img/rho/',
                        'energy': output_dir + 'img/energy/',
                        'fft': output_dir + 'img/fft/',
                        'all_debug': output_dir + 'img/all_debug/',
                        'phase': output_dir + 'img/phase/'}

    return(pic_directories)
