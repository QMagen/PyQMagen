from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from optimizer.optimizer import Optimizer
import copy
import torch
import numpy as np

class BayesianOptimizer(Optimizer):

    def __init__(self, acquisition_func='ei', custom_loss_func=None, num_warm_up=10, **general_arg):
        super(BayesianOptimizer, self).__init__(**general_arg)
        self._optimizer = BayesianOptimization(f=self.eval_loss,
                                               pbounds=self.parameter_space,
                                               verbose=1,
                                               random_state=5)
        self.num_warm_up = num_warm_up
        self.acquisition_func = acquisition_func

    def minimize(self):

        utility = UtilityFunction(kind=self.acquisition_func, kappa=2.5, xi=0.0)

        for i in range(self.n_exp_total):

            if i < self.num_warm_up:
                parameter_space_value = list(self.parameter_space.values())
                next_point = dict(zip(self.parameter_space.keys(),
                                      [np.random.uniform(parameter_space_value[i][0],
                                                         parameter_space_value[i][1])
                                       for i in range(len(parameter_space_value))]))

                loss = self.eval_loss(**next_point)
                loss = -torch.log10(loss)

                self._optimizer.register(params=next_point, target=loss.detach().numpy())

                self.parameter_record.append(copy.deepcopy(next_point))
                self.loss_record[i] = loss.detach().numpy()

            else:

                utility.xi = self.loss_record.std() * 0.01
                next_point = self._optimizer.suggest(utility)

                loss = self.eval_loss(**next_point)
                loss = -torch.log10(loss)

                self._optimizer.register(params=next_point, target=loss.detach().numpy())

                self.parameter_record.append(copy.deepcopy(next_point))
                self.loss_record[i] = loss.detach().numpy()

            print("iteration {}: Loss = {}, Parameter is {}".format(i, loss, next_point))


