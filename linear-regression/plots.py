import numpy as np
import matplotlib.pyplot as plt


def plot_error(xs, err):
    mean = np.mean(xs, axis=1)

    fig, ax = plt.subplots(2, 4)
    ax[0, 0].scatter(xs[:, 0], err)
    ax[0, 1].scatter(xs[:, 1], err)
    ax[0, 2].scatter(xs[:, 2], err)
    ax[0, 3].scatter(xs[:, 3], err)
    ax[1, 0].scatter(xs[:, 4], err)
    ax[1, 1].scatter(xs[:, 5], err)
    ax[1, 2].scatter(xs[:, 6], err)
    ax[1, 3].scatter(mean, err)

    plt.show()


def plot_train_history(history):
    low = np.min(history['loss'] + history['val_loss'])
    high = np.max(history['loss'] + history['val_loss'])
    plt.yscale('log')
    plt.ylim(low, high)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.show()


def plot_log(x, y):
    low = np.min(y)
    high = np.max(y)
    plt.yscale('log')
    plt.ylim(low, high)
    plt.plot(x, y)
    plt.show()


def plot_learning_curve(percentages, curve, marked_percentages):
    low = np.min(curve)
    high = np.max(curve)
    plt.yscale('log')
    plt.ylim(low, high)
    plt.plot(percentages, curve,
             markevery=marked_percentages,
             marker='o')
    plt.show()
