from magen.optimizer.optimizer import OptimizerResult
from magen.optimizer.bayesian_optimizer import BayesianOptimizerResult
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np

def show_landscape3(result, plan_keys, itr=-1):

    assert isinstance(result, BayesianOptimizerResult)
    bounds = [result.parameter_space[plan_keys[0]], result.parameter_space[plan_keys[1]]]

    xs = np.linspace(bounds[0][0], bounds[0][1], 100)
    ys = np.linspace(bounds[1][0], bounds[1][1], 100)
    x_mesh, y_mesh = np.meshgrid(xs, ys)

    predictor = result.BO_record[itr]._gp
    predict_values = np.vstack([predictor.predict(np.array([[xi, yi] for xi in xs]))
                                for yi in ys])
    predict_values = -predict_values

    fig = plt.figure(figsize=[7, 5])

    ax = Axes3D(fig, azim=135, elev=22., proj_type='ortho')
    sf = ax.plot_surface(x_mesh, y_mesh, np.log10(predict_values),
                         rcount=50, ccount=50,
                         cmap=plt.cm.gnuplot_r)

    Z_min = np.log10(predict_values).min() - 0.6 * (np.log10(predict_values).max() -
                                                    np.log10(predict_values).min())

    ctf = ax.contourf(xs, ys, np.log10(predict_values), cmap=plt.cm.gnuplot_r,
                      levels=100, zdir='z', offset=Z_min)

    fig.colorbar(sf, shrink=0.7)

    ax.set_zlim(Z_min, np.log10(predict_values).max())
    ax.set_xlabel(plan_keys[0])
    ax.set_ylabel(plan_keys[1])

    return fig, ax