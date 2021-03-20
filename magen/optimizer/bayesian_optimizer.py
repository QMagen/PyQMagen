from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from .optimizer import Optimizer, OptimizerResult
import copy
import torch
import numpy as np
import dill

class BayesianOptimizer(Optimizer):

    def __init__(self, acquisition_func='ei', record_BO=False,
                 custom_loss_func=None, num_warm_up=10, **general_arg):
        super(BayesianOptimizer, self).__init__(**general_arg)
        self._optimizer = BayesianOptimization(f=self.eval_loss,
                                               pbounds=self.parameter_space,
                                               verbose=1,
                                               random_state=5)
        self.num_warm_up = num_warm_up
        self.record_BO = record_BO
        self.acquisition_func = acquisition_func

    def minimize(self, log_accelerate=True):

        utility = UtilityFunction(kind=self.acquisition_func, kappa=2.5, xi=0.0)
        result = BayesianOptimizerResult()

        if self.record_BO:
            BO_record = []

        for i in range(self.n_exp_total):

            if i < self.num_warm_up:
                parameter_space_value = list(self.parameter_space.values())
                next_point = dict(zip(self.parameter_space.keys(),
                                      [np.random.uniform(parameter_space_value[i][0],
                                                         parameter_space_value[i][1])
                                       for i in range(len(parameter_space_value))]))

                loss = self.eval_loss(**next_point)

                if log_accelerate:
                    loss = -torch.log10(loss)
                else:
                    loss = -loss

                self._optimizer.register(params=next_point, target=loss.detach().numpy())

                self.parameter_record.append(copy.deepcopy(next_point))
                self.loss_record[i] = loss.detach().numpy()

            else:

                utility.xi = self.loss_record.std() * 0.0
                next_point = self._optimizer.suggest(utility)

                loss = self.eval_loss(**next_point)

                if log_accelerate:
                    loss = -torch.log10(loss)
                else:
                    loss = -loss

                self._optimizer.register(params=next_point, target=loss.detach().numpy())

                if self.record_BO:
                    BO_record.append(copy.deepcopy(self._optimizer))

                self.parameter_record.append(copy.deepcopy(next_point))
                self.loss_record[i] = loss.detach().numpy()

            print("iteration {}: Loss = {}, Parameter is {}".format(i + 1, loss, next_point))

        result.parameter_record = self.parameter_record
        result.parameter_space = self.parameter_space

        if log_accelerate:
            result.loss_record = np.power(10, -self.loss_record)
        else:
            result.loss_record = -self.loss_record

        result.best_parameter = self.parameter_record[self.loss_record.argmax()]
        result.best_loss = self.loss_record.min()

        if self.record_BO:
            result.BO_record = BO_record

        print('===================FINISHED=========================')
        print("Best parameter found at iteration {}, with Loss = {}".format(result.loss_record.argmin() + 1,
                                                                            result.best_loss))
        print("{}".format(result.best_parameter))

        return result


class BayesianOptimizerResult(OptimizerResult):

    def __init__(self):
        super(OptimizerResult, self).__init__()
        self.BO_record = None

    def save(self, path):
        fp = open(path, 'wb')
        dill.dump(self, fp)
