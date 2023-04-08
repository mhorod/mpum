'''
Loss functions and their gradients.
'''

from abc import ABC, abstractmethod

import numpy as np

from dataset import *


class LossFunction(ABC):
    @abstractmethod
    def __call__(self, theta: np.ndarray, ds: Dataset):
        pass

    @abstractmethod
    def gradient(self, theta: np.ndarray, ds: Dataset):
        pass


class MSE(LossFunction):
    def __call__(self, theta: np.ndarray, ds: Dataset):
        return sum((theta @ x - y) ** 2 for x, y in ds) / len(ds)

    def gradient(self, theta: np.ndarray, ds: Dataset):
        return 2 * sum(
            (theta @ x - y) * x
            for x, y in ds
        ) / len(ds)


class L1(LossFunction):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, theta: np.ndarray, _: Dataset):
        return self.scale * np.linalg.norm(theta[1:], ord=1)

    def gradient(self, theta, _: Dataset):
        result = self.scale * np.sign(theta)
        result[0] = 0
        return result


class L2(LossFunction):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, theta: np.ndarray, _: Dataset):
        return self.scale * np.linalg.norm(theta[1:], ord=2) ** 2

    def gradient(self, theta, _: Dataset):
        result = self.scale * 2 * theta
        result[0] = 0
        return result


class Sum(LossFunction):
    def __init__(self, *losses):
        self.losses = losses

    def __call__(self, theta: np.ndarray, ds: Dataset):
        return sum(l(theta, ds) for l in self.losses)

    def gradient(self, theta: np.ndarray, ds: Dataset):
        return sum(l.gradient(theta, ds) for l in self.losses)
