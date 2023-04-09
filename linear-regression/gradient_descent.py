'''
Gradient descent algorithm for linear regression.
'''

from dataclasses import dataclass

import numpy as np
from dataset import *
from model import *


@dataclass
class DescentParams:
    model: Model
    train_ds: Dataset
    val_ds: Dataset
    batch_size: int
    epochs: int
    learning_rate: float


def continuous_gradient_descent(params: DescentParams):
    history = {
        'loss': [],
        'val_loss': [],
    }

    for _ in range(params.epochs):
        params.train_ds.shuffle()
        for i in range(0, len(params.train_ds), params.batch_size):
            batch = params.train_ds[i:i + params.batch_size]
            grad = params.model.gradient(batch) * params.learning_rate
            params.model.theta -= grad

        history['loss'].append(params.model.evaluate(params.train_ds))
        if params.val_ds is not None:
            history['val_loss'].append(params.model.evaluate(params.val_ds))

    return history


def make_learning_curve(train_ds, test_ds, percentages, params: DescentParams):
    losses = []
    for p in percentages:
        test_model = params.model.copy().randomize()
        test_params = DescentParams(
            test_model,
            train_ds[:int(len(train_ds) * p)],
            None,
            params.batch_size,
            params.epochs,
            params.learning_rate,
        )
        continuous_gradient_descent(test_params)
        loss = test_model.evaluate(test_ds)
        losses.append(loss)
    return losses


def make_average_learning_curve(train_ds, test_ds, percentages, attempts, params: DescentParams):
    losses = [
        make_learning_curve(train_ds, test_ds, percentages, params)
        for _ in range(attempts)
    ]
    return np.mean(losses, axis=0)
