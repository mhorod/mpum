import numpy as np

from regression.dataset import *


class Model:
    def __init__(self, loss, base_functions, theta=None):
        self.loss = loss
        self.base_functions = base_functions
        self.theta = theta if theta is not None else np.zeros(self.features())

    def prepare_dataset(self, ds):
        xs = np.array([[f(x) for f in self.base_functions] for x in ds.xs])
        ys = ds.ys
        return Dataset(xs, ys)

    def features(self):
        return len(self.base_functions)

    def __call__(self, xs):
        return xs @ self.theta

    def evaluate(self, ds):
        return self.loss(self.theta, ds)

    def gradient(self, ds):
        return self.loss.gradient(self.theta, ds)

    def randomize(self):
        self.theta = np.random.default_rng().random(self.features()) * 2 - 1
        self.theta *= 0.1
        return self

    def copy(self):
        return Model(self.loss, self.base_functions.copy(), self.theta.copy())
