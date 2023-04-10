import numpy as np
import matplotlib.pyplot as plt


def plot_by_x(xs, err, color=None, filename=None):
    plt.clf()

    mean = np.mean(xs, axis=1)

    fig, ax = plt.subplots(2, 4)
    fig.tight_layout()

    ax[0, 0].scatter(xs[:, 0], err, c=color)
    ax[0, 0].set_title('x1')

    ax[0, 1].scatter(xs[:, 1], err, c=color)
    ax[0, 1].set_title('x2')

    ax[0, 2].scatter(xs[:, 2], err, c=color)
    ax[0, 2].set_title('x3')

    ax[0, 3].scatter(xs[:, 3], err, c=color)
    ax[0, 3].set_title('x4')

    ax[1, 0].scatter(xs[:, 4], err, c=color)
    ax[1, 0].set_title('x5')

    ax[1, 1].scatter(xs[:, 5], err, c=color)
    ax[1, 1].set_title('x6')

    ax[1, 2].scatter(xs[:, 6], err, c=color)
    ax[1, 2].set_title('x7')

    ax[1, 3].scatter(mean, err, c=color)
    ax[1, 3].set_title('mean')

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
    else:
        plt.show()


def plot_train_history(history, filename=None):
    plt.clf()
    plt.tight_layout()

    low = np.min(history['loss'] + history['val_loss'])
    high = np.max(history['loss'] + history['val_loss'])
    plt.yscale('log')
    plt.ylim(low, high)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()


def plot_learning_curve(percentages, curve, marked_percentages, filename=None):
    plt.clf()
    plt.tight_layout()

    low = np.min(curve)
    high = np.max(curve)
    plt.yscale('log')
    plt.ylim(low, high)
    plt.plot(percentages, curve,
             markevery=marked_percentages,
             marker='o',
             )

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()
