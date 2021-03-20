import numpy as np
import json
from magen.aux_func import calcul_loss
from magen.aux_func import get_cutoff_t

class Optimizer:

    def __init__(self, n_exp_total, parameter_space, solver, model, exp_thermal_data,
                 target_obs=None, T_cut=None):
        self.n_exp_total = n_exp_total
        self.parameter_space = parameter_space
        self.solver = solver
        self.model = model
        self.exp_thermal_data = exp_thermal_data

        if target_obs is None:
            self.target_obs = ['C']
        else:
            self.target_obs = target_obs



        if T_cut is None:
            self.T_cut = dict(zip(self.target_obs, [len(exp_thermal_data.T) for _ in range(len(self.target_obs))]))
        else:
            self.T_cut = get_cutoff_t(self.target_obs, self.exp_thermal_data, shift=T_cut)

        self.parameter_record = []
        self.loss_record = np.zeros(n_exp_total)

    def eval_loss(self, **parameter_point):

        interactions = self.model.generate_interactions(**parameter_point)
        simulated_thermal_data = self.solver.forward(interactions=interactions,
                                                     T=self.exp_thermal_data.T)
        loss = calcul_loss(obs_real=self.exp_thermal_data,
                           obs_iter=simulated_thermal_data,
                           target_obs=self.target_obs,
                           T_cut=self.T_cut)

        return loss

class OptimizerResult:

    def __init__(self):
        self.parameter_record = None
        self.parameter_space = None
        self.loss_record = None
        self.best_parameter = None
        self.best_loss = None
        pass

    def save(self, path):
        fp = open(path, 'wt')
        json.dump(self, fp)
