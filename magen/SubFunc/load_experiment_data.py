import numpy as np
import torch
from magen.adlib import ThObs

def load_experiment_data(experimental_files):
    T = np.loadtxt(experimental_files['T'], delimiter=',')
    exp_thermal_data = ThObs(T)
    for thermal_obs_name in experimental_files.keys():
        data_temp = np.loadtxt(experimental_files[thermal_obs_name], delimiter=',')
        setattr(exp_thermal_data, thermal_obs_name, torch.tensor(data_temp))

    return exp_thermal_data


