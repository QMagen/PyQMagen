from .bayesian_optimizer import BayesianOptimizer
from SubFunc import calcul_loss
from .optimizer import Optimizer
import numpy as np


# class Optimizer:
#
#     def __init__(self, n_exp_total, parameter_space, solver, model, exp_thermal_data):
#         self.n_exp_total = n_exp_total
#         self.parameter_space = parameter_space
#         self.solver = solver
#         self.model = model
#         self.exp_thermal_data = exp_thermal_data
#
#         self.parameter_record = []
#         self.loss_record = np.zeros(n_exp_total)
#
#     def eval_loss(self, **parameter_point):
#
#         interactions = self.model.generate_interactions(**parameter_point)
#         simluated_thermal_data = self.solver.forward(interactions=interactions,
#                                                      T=self.exp_thermal_data.T)
#         loss = calcul_loss(obs_real=self.exp_thermal_data,
#                            obs_iter=simluated_thermal_data,
#                            target_obs=[],
#                            T_cut=None)
#
#         return loss
