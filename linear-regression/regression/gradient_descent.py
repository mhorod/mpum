'''
Gradient descent algorithm for linear regression.
'''

from dataclasses import dataclass

import numpy as np
from regression.dataset import *
from regression.model import *


@dataclass
class DescentParams:
    model: Model
    train_ds: Dataset
    val_ds: Dataset
    batch_size: int
    epochs: int
    learning_rate: float


@dataclass
class DescentMetaParams:
    batch_size: int
    epochs: int
    learning_rate: float

    def into_descent_params(self, model: Model, train_ds: Dataset, val_ds: Dataset):
        return DescentParams(
            model,
            train_ds,
            val_ds,
            self.batch_size,
            self.epochs,
            self.learning_rate,
        )


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


def continuous_gradient_descent_with_momentum(params: DescentParams):
    history = {
        'loss': [],
        'val_loss': [],
    }

    gs2 = 0
    for _ in range(params.epochs):
        params.train_ds.shuffle()
        for i in range(0, len(params.train_ds), params.batch_size):
            batch = params.train_ds[i:i + params.batch_size]
            grad = params.model.gradient(batch)
            gs2 = gs2 * 0.9 + 0.1 * grad ** 2
            grad /= np.sqrt(gs2)
            params.model.theta -= grad * params.learning_rate

        history['loss'].append(params.model.evaluate(params.train_ds))
        if params.val_ds is not None:
            history['val_loss'].append(params.model.evaluate(params.val_ds))

    return history


def coordinate_gradient_descent(params: DescentParams):
    history = {
        'loss': [],
        'val_loss': [],
    }

    for _ in range(params.epochs):
        params.train_ds.shuffle()
        j = np.random.randint(0, len(params.model.theta))
        for i in range(0, len(params.train_ds), params.batch_size):
            batch = params.train_ds[i:i + params.batch_size]
            grad = params.model.gradient(batch) * params.learning_rate
            params.model.theta[j] -= grad[j]

        history['loss'].append(params.model.evaluate(params.train_ds))
        if params.val_ds is not None:
            history['val_loss'].append(params.model.evaluate(params.val_ds))

    return history


def make_learning_curve(train_ds, test_ds, percentages, params: DescentParams):
    losses = []
    for p in percentages:
        print(f"Training on {p * 100:.2f}% of data")
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
