import numpy as np

import pickle


class Predictor:
    def __init__(self, normalization, theta, base_functions):
        self.x_mean, self.x_std = normalization[0]
        self.y_mean, self.y_std = normalization[1]
        self.theta = theta
        self.base_functions = base_functions

    def __call__(self, xs):
        xs = (xs - self.x_mean) / self.x_std
        xs = np.array([[f(x) for f in self.base_functions] for x in xs])
        ys = xs @ self.theta
        ys = ys * self.y_std + self.y_mean
        return ys


def export_predictor(predictor, path):
    with open(path, 'wb') as f:
        pickle.dump(predictor, f)


def load_predictor(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
