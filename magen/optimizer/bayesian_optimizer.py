from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from .optimizer import Optimizer, OptimizerResult
import copy
import torch
import numpy as np
import dill


class BayesianOptimizer(Optimizer):

    def __init__(self, exp_thermal_data,
                 n_exp_total,
                 parameter_space,
                 solver,
                 model,
                 acquisition_func='ei', record_BO=False,
                 custom_loss_func=None, num_warm_up=10,
                 target_obs=None, T_cut=None,
                 **args):

        super(BayesianOptimizer, self).__init__(exp_thermal_data=exp_thermal_data,
                                                n_exp_total=n_exp_total,
                                                parameter_space=parameter_space,
                                                solver=solver,
                                                model=model,
                                                T_cut=T_cut,
                                                target_obs=target_obs
                                                )

        self._optimizer = BayesianOptimization(f=self.eval_loss,
                                               pbounds=self.parameter_space,
                                               verbose=1,
                                               random_state=5)
        self.num_warm_up = num_warm_up
        self.record_BO = record_BO
        self.acquisition_func = acquisition_func

        if custom_loss_func:
            self.eval_loss = custom_loss_func

    def minimize(self, log_accelerate=True):

        utility = UtilityFunction(kind=self.acquisition_func, kappa=2.5, xi=0.0)
        result = BayesianOptimizerResult()

        aux = '|{{:=^{}s}}|'.format(len(self.parameter_space) * 11 + 21)
        print(aux.format('Optimization Start'))
        output_line = '|{:^10s}|{:^10s}|'.format('Itr', 'Loss')
        for i in range(len(self.parameter_space)):
            output_line += '{:^10s}|'.format(list(self.parameter_space.keys())[i])
        print(output_line)

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

            if log_accelerate:
                loss_to_save = np.power(10, -loss.detach().numpy())
            else:
                loss_to_save = -loss.detach().numpy()

            self.loss_record[i] = loss_to_save

            output_line = '|{:^10d}|{:^10.2e}|'.format(i, loss_to_save)
            for j in range(len(self.parameter_space)):
                output_line += '{:^10.5f}|'.format(next_point[list(self.parameter_space.keys())[j]])
            print(output_line)

        result.parameter_record = self.parameter_record
        result.parameter_space = self.parameter_space
        result.BO_record = BO_record
        # if log_accelerate:
        #     result.loss_record = np.power(10, -self.loss_record)
        # else:
        #     result.loss_record = -self.loss_record

        result.loss_record = self.loss_record
        result.best_parameter = self.parameter_record[self.loss_record.argmin()]
        result.best_loss = self.loss_record.min()



        print(aux.format('Optimization Finised'))

        output_line = '|{:^10s}|{:^10s}|'.format('Best', 'Loss')
        for i in range(len(self.parameter_space)):
            output_line += '{:^10s}|'.format(list(self.parameter_space.keys())[i])
        print(output_line)

        output_line = '|{:^10d}|{:^10.2e}|'.format(result.loss_record.argmin() + 1, result.best_loss)
        for j in range(len(self.parameter_space)):
            output_line += '{:^10.5f}|'.format(result.best_parameter[list(self.parameter_space.keys())[j]])
        print(output_line)

        # print("Best parameter found at iteration {}, with Loss = {}".format(result.loss_record.argmin() + 1,
        #                                                                     result.best_loss))
        # print("{}".format(result.best_parameter))

        return result


class BayesianOptimizerResult(OptimizerResult):

    def __init__(self):
        super(OptimizerResult, self).__init__()
        self.BO_record = None

    def save(self, path):
        fp = open(path, 'wb')
        dill.dump(self, fp)
