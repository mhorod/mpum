'''
Gradient descent algorithm for linear regression.
'''

import numpy as np
from dataset import *


def continuous_gradient_descent(loss, theta, train_ds, learning_rate, epochs, val_ds=None,
                                batch_size=1):
    history = {
        'loss': [],
        'val_loss': [],
    }

    for _ in range(epochs):
        train_ds.shuffle()
        for i in range(0, len(train_ds), batch_size):
            batch = train_ds[i:i + batch_size]
            grad = loss.gradient(theta, batch) * learning_rate
            theta -= grad

        history['loss'].append(loss(theta, train_ds))
        if val_ds is not None:
            history['val_loss'].append(loss(theta, val_ds))

    return theta, history


def make_learning_curve(train_ds, percentages, model):
    losses = []
    for p in percentages:
        ds = train_ds[:int(len(train_ds) * p)]
        _, history = model(ds)
        losses.append(history['loss'][-1])
    return losses


def make_model(loss, learning_rate, epochs, batch_size=1):
    def model(train_ds, val_ds=None):
        theta = np.random.default_rng().random(len(train_ds.xs[0]))
        return continuous_gradient_descent(loss, theta, train_ds, learning_rate, epochs, val_ds, batch_size)
    return model
