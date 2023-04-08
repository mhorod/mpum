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
        k = 0
        grad = np.zeros(theta.shape)
        train_ds.shuffle()
        for x, y in train_ds:
            grad += learning_rate * loss.gradient(theta, Dataset([x], [y]))
            k += 1
            if k % batch_size == 0:
                theta -= grad / batch_size
                grad = np.zeros(theta.shape)

        history['loss'].append(loss(theta, train_ds))
        if val_ds is not None:
            history['val_loss'].append(loss(theta, val_ds))
    return theta, history
