from .bayesian_optimizer import BayesianOptimizer
from .optimizer import Optimizer
import json
import dill


def load_optimize_result(path):
    fp = open(path, 'rt')
    return json.load(fp)


def load_optimize_result_bayesian(path):
    fp = open(path, 'rb')
    return dill.load(fp)