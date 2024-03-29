from qmagen.optimizer.optimizer import OptimizerResult
from qmagen.optimizer.bayesian_optimizer import BayesianOptimizerResult
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
import warnings
import numpy as np


def show_landscape_gp(result, plan_keys, itr=-1, fix_param=None, log_scale=False):

    """
    :param result: a OptimizerResult object
    :param plan_keys: list,
        should contain two names of parameters, e.g. ['Jz', 'Jx']
    :param itr: int, default -1,
        the landscape at which iteration to plot
    :param fix_param: dict or None, default: None
        specify the parameter value of parameter not in plan_keys, e.g. {'Jy': 1}
        if None, it will be automatically set to optimal parameter values
    :param log_scale: bool, default: False,
        if True, the colorbar will be in logarithmic scale
    :return: None
    """

    assert isinstance(result, OptimizerResult)
    bounds = [result.parameter_space[plan_keys[0]], result.parameter_space[plan_keys[1]]]

    xs = np.linspace(bounds[0][0], bounds[0][1], 100)
    ys = np.linspace(bounds[1][0], bounds[1][1], 100)

    def get_param(xi, yi):
        param_names = list(result.parameter_space.keys())
        p_point = np.zeros(len(result.parameter_space))
        if len(plan_keys) == len(result.parameter_space):
            return np.array([xi, yi])
        elif fix_param:
            i = 0
            for param_name in param_names:
                if param_name == plan_keys[0]:
                    p_point[i] = xi
                elif param_name == plan_keys[1]:
                    p_point[i] = yi
                else:
                    p_point[i] = fix_param[param_name]
                i += 1
        else:
            i = 0
            for param_name in param_names:
                if param_name == plan_keys[0]:
                    p_point[i] = xi
                elif param_name == plan_keys[1]:
                    p_point[i] = yi
                else:
                    p_point[i] = result.BO_record[itr].max['params'][param_name]
                i += 1
        return p_point

    X = np.vstack([np.array([result.parameter_record[i][list(result.parameter_space.keys())[j]]
                             for j in range(len(result.parameter_space))]) for i in range(len(result.loss_record))])

    Y = result.loss_record

    GP = GaussianProcessRegressor(
        kernel=Matern(nu=2.5, length_scale_bounds=(1e-05, 1000)),
        alpha=1e-6,
        optimizer='fmin_l_bfgs_b',
        normalize_y=True,
        n_restarts_optimizer=200,
    )

    # GP.fit(X, np.power(10, -Y))
    GP.fit(X, Y)

    predict_values = np.vstack([GP.predict(np.array([get_param(xi, yi) for xi in xs]))
                                for yi in ys])

    fig, ax = plt.subplots(figsize=[5, 5])

    predict_values[np.where(predict_values < 0)] = 1e-3
    # predict_values = -predict_values

    if log_scale:
        ctf = ax.contourf(xs, ys, predict_values, cmap=plt.cm.gnuplot_r,
                          norm=colors.LogNorm(vmin=predict_values.min(), vmax=predict_values.max()),
                          levels=np.power(10, np.linspace(np.log10(predict_values.min()),
                                                          np.log10(predict_values.max()), 100)))
    else:
        ctf = ax.contourf(xs, ys, predict_values, cmap=plt.cm.gnuplot_r,
                          levels=100)

    fig.colorbar(ctf)

    ax.set_xlabel(plan_keys[0])
    ax.set_ylabel(plan_keys[1])

    return fig, ax


def show_landscape(result, plan_keys, itr=-1, fix_param=None, log_scale=False):

    """
    :param result: a OptimizerResult object
    :param plan_keys: list,
        should contain two names of parameters, e.g. ['Jz', 'Jx']
    :param itr: int, default -1,
        the landscape at which iteration to plot
    :param fix_param: dict or None, default: None
        specify the parameter value of parameter not in plan_keys, e.g. {'Jy': 1}
        if None, it will be automatically set to optimal parameter values
    :param log_scale: bool, default: False,
        if True, the colorbar will be in logarithmic scale
    :return: None
    """

    if log_scale:
        warn_message = 'Using log scale might cause numerical errors when the parameter space ' \
                       'is not fully explored (Gaussian process might predict negative values'
        warnings.warn(warn_message)
    assert isinstance(result, BayesianOptimizerResult)
    bounds = [result.parameter_space[plan_keys[0]], result.parameter_space[plan_keys[1]]]

    def get_param(xi, yi):
        param_names = list(result.parameter_space.keys())
        p_point = np.zeros(len(result.parameter_space))
        if len(plan_keys) == len(result.parameter_space):
            return np.array([xi, yi])
        elif fix_param:
            i = 0
            for param_name in param_names:
                if param_name == plan_keys[0]:
                    p_point[i] = xi
                elif param_name == plan_keys[1]:
                    p_point[i] = yi
                else:
                    p_point[i] = fix_param[param_name]
                i += 1
        else:
            i = 0
            for param_name in param_names:
                if param_name == plan_keys[0]:
                    p_point[i] = xi
                elif param_name == plan_keys[1]:
                    p_point[i] = yi
                else:
                    p_point[i] = result.BO_record[itr].max['params'][param_name]
                i += 1
        return p_point

    xs = np.linspace(bounds[0][0], bounds[0][1], 100)
    ys = np.linspace(bounds[1][0], bounds[1][1], 100)

    predictor = result.BO_record[itr]._gp

    predict_values = np.vstack([predictor.predict(np.array([get_param(xi, yi) for xi in xs]))
                                for yi in ys])

    fig, ax = plt.subplots(figsize=[5, 5])

    # ctf = ax.contourf(xs, ys, predict_values, cmap=plt.cm.gnuplot,
    #                   levels=100)
    predict_values = -predict_values

    if log_scale:
        ctf = ax.contourf(xs, ys, predict_values, cmap=plt.cm.gnuplot_r,
                          norm=colors.LogNorm(vmin=predict_values.min(), vmax=predict_values.max()),
                          levels=np.power(10, np.linspace(np.log10(predict_values.min()),
                                                          np.log10(predict_values.max()), 100)))
    else:
        ctf = ax.contourf(xs, ys, predict_values, cmap=plt.cm.gnuplot_r,
                          levels=100)

    fig.colorbar(ctf)

    ax.set_xlabel(plan_keys[0])
    ax.set_ylabel(plan_keys[1])

    return fig, ax