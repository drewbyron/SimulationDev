import spec_tools.analysis_functions.sideband_power_calc as spc

import os
import os.path
import json

import sys

home_dir = os.getcwd()
user_config_file = home_dir + "/user_config.json"

if os.path.exists(user_config_file):
    print("User config file exists, loading...")
    
    with open(user_config_file,"r") as read_file:
        try:
            user_dict = json.load(read_file)
            filename = user_dict["power_analysis_config"]
            if not filename[0] == "/":
                data_file = home_dir + "/" + filename
            else:
                data_file = home_dir + filename
        except:
            print('Loaded file "{}" does not contain a valid user config dictionary'.format(filename))
            sys.exit()
    
else:
    #choose spectra build config file in config_files/
    filename ="power_simulation_files/power_simulation.json"

    #loading config file
    data_file = os.getcwd() + "/config_files/" + filename

if not os.path.exists(data_file):
    print('File "{}" does not exist'.format(data_file))
    print()
    sys.exit()
    
with open(data_file,"r") as read_file:
    try:
        config_dict = json.load(read_file)
    except:
        print('Loaded file "{}" does not contain a valid spectra analysis config dictionary'.format(filename))
        sys.exit()


simulation_dict = config_dict["simulation_dict"]
analysis_dict = config_dict["analysis_dict"]

plot_sideband_powers_test = config_dict["plot_sideband_powers_test"]

run_power_simulation = config_dict["run_power_simulation"]
plot_generated_beta_path = config_dict["plot_generated_beta_path"]
plot_sideband_powers = config_dict["plot_sideband_powers"]

analyze_power_simulations = config_dict["analyze_power_simulations"]
analyze_power_slopes = config_dict["analyze_power_slopes"]
check_sidebands = config_dict["check_sidebands"]


if plot_sideband_powers_test == True:

    spc.plot_sideband_powers(20e9,1,88,1e-3)

if run_power_simulation == True:

    spc.run_power_simulations(simulation_dict,
        plot_generated_beta_path,
        plot_sideband_powers)

if analyze_power_simulations == True:

    spc.analyze_power_simulations(analysis_dict,check_sidebands)
    
if analyze_power_slopes == True:

    spc.analyze_power_slopes(analysis_dict,check_sidebands)

    

