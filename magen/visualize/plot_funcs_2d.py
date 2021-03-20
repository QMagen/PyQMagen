from magen.optimizer.optimizer import OptimizerResult
from magen.optimizer.bayesian_optimizer import BayesianOptimizerResult
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

import numpy as np


def show_landscape_gp(result, plan_keys):

    assert isinstance(result, OptimizerResult)
    bounds = [result.parameter_space[plan_keys[0]], result.parameter_space[plan_keys[1]]]

    xs = np.linspace(bounds[0][0], bounds[0][1], 100)
    ys = np.linspace(bounds[1][0], bounds[1][1], 100)

    X = np.vstack([np.array([result.parameter_record[i][plan_keys[0]],
                             result.parameter_record[i][plan_keys[1]]]) for i in range(len(result.loss_record))])

    Y = result.loss_record

    GP = GaussianProcessRegressor(
        kernel=Matern(nu=2.5, length_scale_bounds=(1e-05, 1000)),
        alpha=1e-6,
        optimizer='fmin_l_bfgs_b',
        # optimizer=None,
        normalize_y=True,
        n_restarts_optimizer=100,
    )

    # GP.fit(X, np.power(10, -Y))
    GP.fit(X, Y)

    predict_values = np.vstack([GP.predict(np.array([[xi, yi] for xi in xs]))
                                for yi in ys])

    fig, ax = plt.subplots(figsize=[5, 5])

    predict_values[np.where(predict_values < 0)] = 1e-3
    # predict_values = -predict_values

    ctf = ax.contourf(xs, ys, predict_values, cmap=plt.cm.gnuplot_r,
                      norm=colors.LogNorm(vmin=predict_values.min(), vmax=predict_values.max()),
                      levels=np.power(10, np.linspace(np.log10(predict_values.min()),
                                                      np.log10(predict_values.max()), 100)))

    # ctf = ax.contourf(xs, ys, predict_values, cmap=plt.cm.gnuplot,
    #                   levels=100)

    # ctf = ax.contourf(xs, ys, np.log10(predict_values), cmap=plt.cm.gnuplot_r,
    #                   levels=100)
    fig.colorbar(ctf)

    ax.set_xlabel(plan_keys[0])
    ax.set_ylabel(plan_keys[1])

    return fig, ax

def show_landscape(result, plan_keys, itr=-1):

    assert isinstance(result, BayesianOptimizerResult)
    bounds = [result.parameter_space[plan_keys[0]], result.parameter_space[plan_keys[1]]]

    xs = np.linspace(bounds[0][0], bounds[0][1], 100)
    ys = np.linspace(bounds[1][0], bounds[1][1], 100)

    predictor = result.BO_record[itr]._gp
    predict_values = np.vstack([predictor.predict(np.array([[xi, yi] for xi in xs]))
                                for yi in ys])

    fig, ax = plt.subplots(figsize=[5, 5])

    # ctf = ax.contourf(xs, ys, predict_values, cmap=plt.cm.gnuplot,
    #                   levels=100)
    predict_values = -predict_values
    ctf = ax.contourf(xs, ys, predict_values, cmap=plt.cm.gnuplot_r,
                      norm=colors.LogNorm(vmin=predict_values.min(), vmax=predict_values.max()),
                      levels=np.power(10, np.linspace(np.log10(predict_values.min()),
                                                      np.log10(predict_values.max()), 100)))

    fig.colorbar(ctf)

    ax.set_xlabel(plan_keys[0])
    ax.set_ylabel(plan_keys[1])

    return fig, ax