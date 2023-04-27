import numpy as np

from common import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gradient_descent(ds, learning_rate, max_iter, val_ds=None, l2=0, save_history=False):
    history = {'val_ll': [], 'val_acc': []}
    theta = np.random.default_rng().random(FEATURES) * 0.1
    intercept = np.random.default_rng().random() * 0.1
    for _ in range(max_iter):
        for row in ds.values:
            x = row[:FEATURES]
            y = row[FEATURES]
            y = 1 if y == MALIGNANT else 0
            d = sigmoid(np.dot(theta, x) + intercept)
            grad = (y - d) * x
            regularization = 2 * l2 * theta
            theta += learning_rate * grad - regularization
            intercept += learning_rate * (y - d)

        if save_history and val_ds is not None:
            history['val_ll'].append(
                log_likelihood(val_ds, (theta, intercept)))
            history['val_acc'].append(
                measure_accuracy(val_ds, (theta, intercept)))

    if save_history:
        return theta, intercept, history
    else:
        return theta, intercept


def predict_one(x, theta):
    theta, intercept = theta
    p = sigmoid(np.dot(theta, x) + intercept)
    if p > 0.5:
        return MALIGNANT
    else:
        return BENIGN


def measure_accuracy(ds, theta):
    correct = 0
    for row in ds.values:
        x = row[:FEATURES]
        y = row[FEATURES]
        y_hat = predict_one(x, theta)
        if y == y_hat:
            correct += 1

    return correct / len(ds)


def log_likelihood(ds, theta):
    l = 0
    theta, intercept = theta
    for row in ds.values:
        p = 1
        x = row[:FEATURES]
        y = row[FEATURES]

        if y == MALIGNANT:
            p = sigmoid(np.dot(theta, x) + intercept)
        else:
            p = 1 - sigmoid(np.dot(theta, x) + intercept)
        l += np.log(p)

    return l / len(ds)
