from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from common import *
import logistic_regression
import naive_bayes


import sys


@dataclass
class Stats:
    train_acc: float
    test_acc: float

    train_log_likelihood: float
    test_log_likelihood: float

    def __str__(self):
        return f"Train Accuracy: {self.train_acc}\nTest Accuracy: {self.test_acc}\nTrain Log Likelihood: {self.train_log_likelihood}\nTest Log Likelihood: {self.test_log_likelihood}"


def logistic_regression_epochs_stats(train_ds, test_ds, iterations=5):
    epochs = 2000
    test_ll = np.zeros(epochs)
    test_acc = np.zeros(epochs)

    for i in range(iterations):
        print("Iteration: ", i)
        train_ds = train_ds.sample(frac=1)
        theta, intercept, history = logistic_regression.gradient_descent(
            train_ds, 0.0005, epochs, test_ds, l2=0, save_history=True)
        test_ll += history['val_ll']
        test_acc += history['val_acc']

    test_ll /= iterations
    test_acc /= iterations

    # plot on two figures
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Logistic Regression')
    ax1.plot(range(epochs), test_ll)
    ax1.set_title('Log Likelihood')
    ax2.plot(range(epochs), test_acc)
    ax2.set_title('Accuracy')
    fig.tight_layout()
    plt.show()


def naive_bayes_stats(train_ds, test_ds, sample_size=None, iterations=5):
    stats = Stats(0, 0, 0, 0)
    if sample_size is None:
        sample_size = len(train_ds)
    for _ in range(iterations):
        ds = train_ds.sample(sample_size)
        classifier = naive_bayes.make_naive_bayes(ds)
        stats.train_acc += naive_bayes.measure_accuracy(ds, classifier)
        stats.test_acc += naive_bayes.measure_accuracy(test_ds, classifier)
        stats.train_log_likelihood += naive_bayes.log_likelihood(
            ds, classifier)
        stats.test_log_likelihood += naive_bayes.log_likelihood(
            test_ds, classifier)

    stats.train_acc /= iterations
    stats.test_acc /= iterations
    stats.test_log_likelihood /= iterations
    stats.train_log_likelihood /= iterations
    return stats


def logistic_regression_stats(train_ds, test_ds, sample_size=None, iterations=5):
    stats = Stats(0, 0, 0, 0)
    if sample_size is None:
        sample_size = len(train_ds)
    for _ in range(iterations):
        ds = train_ds.sample(sample_size)
        theta = logistic_regression.gradient_descent(ds, 0.0005, 500)
        stats.train_acc += logistic_regression.measure_accuracy(ds, theta)
        stats.test_acc += logistic_regression.measure_accuracy(test_ds, theta)
        stats.train_log_likelihood += logistic_regression.log_likelihood(
            ds, theta)
        stats.test_log_likelihood += logistic_regression.log_likelihood(
            test_ds, theta)

    stats.train_acc /= iterations
    stats.test_acc /= iterations
    stats.test_log_likelihood /= iterations
    stats.train_log_likelihood /= iterations
    return stats


def main_comparison(train_ds, test_ds, iterations=5):
    nb_stats = []
    lr_stats = []

    f1 = [0.01, 0.02, 0.03, 0.125, 0.625, 1]
    f2 = [0.1 * i for i in range(1, 11)]

    fractions = list(sorted(set(f1 + f2)))

    for frac in fractions:
        print("Fraction: ", frac)
        size = int(len(train_ds) * frac)
        ds = train_ds.sample(size)
        nb_stats.append(naive_bayes_stats(ds, test_ds, iterations))
        lr_stats.append(logistic_regression_stats(ds, test_ds, iterations))

    # plot all 4 stats on separate graphs
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].plot(fractions, [s.train_acc for s in nb_stats],
                  label="Naive Bayes")
    ax[0, 0].plot(fractions, [s.train_acc for s in lr_stats],
                  label="Logistic Regression")
    ax[0, 0].set_title("Train Accuracy")
    ax[0, 0].set_xlabel("Fraction of training set")
    ax[0, 0].set_ylabel("Accuracy")
    ax[0, 0].legend()

    ax[0, 1].plot(fractions, [s.test_acc for s in nb_stats],
                  label="Naive Bayes")
    ax[0, 1].plot(fractions, [s.test_acc for s in lr_stats],
                  label="Logistic Regression")
    ax[0, 1].set_title("Test Accuracy")
    ax[0, 1].set_xlabel("Fraction of training set")
    ax[0, 1].set_ylabel("Accuracy")
    ax[0, 1].legend()

    ax[1, 0].plot(
        fractions, [s.train_log_likelihood for s in nb_stats], label="Naive Bayes")
    ax[1, 0].plot(fractions, [s.train_log_likelihood for s in lr_stats],
                  label="Logistic Regression")
    ax[1, 0].set_title("Train Log Likelihood")
    ax[1, 0].set_xlabel("Fraction of training set")
    ax[1, 0].set_ylabel("Log Likelihood")
    ax[1, 0].legend()

    ax[1, 1].plot(fractions, [s.test_log_likelihood for s in nb_stats],
                  label="Naive Bayes")
    ax[1, 1].plot(fractions, [s.test_log_likelihood for s in lr_stats],
                  label="Logistic Regression")
    ax[1, 1].set_title("Test Log Likelihood")
    ax[1, 1].set_xlabel("Fraction of training set")
    ax[1, 1].set_ylabel("Log Likelihood")
    ax[1, 1].legend()

    # set equal y limits for all graphs
    min_train_acc = min([s.train_acc for s in nb_stats + lr_stats])
    max_train_acc = max([s.train_acc for s in nb_stats + lr_stats])
    min_test_acc = min([s.test_acc for s in nb_stats + lr_stats])
    max_test_acc = max([s.test_acc for s in nb_stats + lr_stats])

    min_train_ll = min([s.train_log_likelihood for s in nb_stats + lr_stats])
    min_test_ll = min([s.test_log_likelihood for s in nb_stats + lr_stats])

    min_acc = min(min_train_acc, min_test_acc)
    max_acc = max(max_train_acc, max_test_acc)
    min_ll = min(min_train_ll, min_test_ll)
    max_ll = 0

    ax[0, 0].set_ylim(min_acc, max_acc)
    ax[0, 1].set_ylim(min_acc, max_acc)
    ax[1, 0].set_ylim(min_ll, max_ll)
    ax[1, 1].set_ylim(min_ll, max_ll)

    plt.tight_layout()
    plt.show()


def display_naive_bayes_info(train_ds, test_ds):
    stats = naive_bayes_stats(train_ds, test_ds, iterations=5)

    bayes = naive_bayes.make_naive_bayes(train_ds)
    print("Prior probabilities: ", bayes.y_phi)

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].matshow(bayes.x_phis[0], vmin=0, vmax=1)
    ax[0].set_title("Benign")
    ax[0].set_ylabel("Feature")
    ax[0].set_xlabel("Value")

    ax[1].matshow(bayes.x_phis[1], vmin=0, vmax=1)
    ax[1].set_title("Malignant")
    ax[1].set_ylabel("Feature")
    ax[1].set_xlabel("Value")

    ax[0].set_xticks(range(0, 10))
    ax[0].set_xticklabels(range(1, 11))
    ax[0].set_yticks(range(0, 9))

    ax[1].set_xticks(range(0, 10))
    ax[1].set_xticklabels(range(1, 11))
    ax[1].set_yticks(range(0, 9))

    fig.subplots_adjust(right=0.8)
    fig.colorbar(ax[0].images[0], ax=ax.ravel().tolist(),
                 fraction=0.046, pad=0.04)

    plt.show()

    for i in range(FEATURES):
        s0 = sum(bayes.x_phis[0][i])
        s1 = sum(bayes.x_phis[0][i])
        print("Feature ", i, ":", s0, s1)


def test_on_extreme_dataset(benign, malignant):
    test_ds = pd.concat([benign[2:], malignant[1:]])
    train_ds = pd.concat([benign[:2], malignant[:1]])

    print("Train set size: ", len(train_ds), "of which", len(
        train_ds[train_ds[FEATURES] == BENIGN]), "benign")
    print("Test set size: ", len(test_ds), "of which", len(
        test_ds[test_ds[FEATURES] == BENIGN]), "benign")
    print()

    print("NB: ")
    print(naive_bayes_stats(train_ds, test_ds, iterations=20))
    print()

    print("LR: ")
    print(logistic_regression_stats(train_ds, test_ds, iterations=20))


def test_on_extreme_datasets(train_ds, test_ds):
    nb_stats = []
    lr_stats = []

    xs = [i for i in range(1, 20)]
    for i in xs:
        print("Testing on ", i, "samples")
        nb_stats.append(naive_bayes_stats(
            train_ds, test_ds, sample_size=i, iterations=20))
        lr_stats.append(logistic_regression_stats(
            train_ds, test_ds, sample_size=i, iterations=20))

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(2):
        for j in range(2):
            ax[i, j].set_xlim(xs[0], xs[-1])
            ax[i, j].set_xticks(xs)

    ax[0, 0].plot(xs, [s.train_acc for s in nb_stats],
                  label="Naive Bayes")
    ax[0, 0].plot(xs, [s.train_acc for s in lr_stats],
                  label="Logistic Regression")
    ax[0, 0].set_title("Train Accuracy")
    ax[0, 0].set_xlabel("Size of training set")
    ax[0, 0].set_ylabel("Accuracy")
    ax[0, 0].legend()

    ax[0, 1].plot(xs, [s.test_acc for s in nb_stats],
                  label="Naive Bayes")
    ax[0, 1].plot(xs, [s.test_acc for s in lr_stats],
                  label="Logistic Regression")
    ax[0, 1].set_title("Test Accuracy")
    ax[0, 1].set_xlabel("Size of training set")
    ax[0, 1].set_ylabel("Accuracy")
    ax[0, 1].legend()

    ax[1, 0].plot(
        xs, [s.train_log_likelihood for s in nb_stats], label="Naive Bayes")
    ax[1, 0].plot(xs, [s.train_log_likelihood for s in lr_stats],
                  label="Logistic Regression")
    ax[1, 0].set_title("Train Log Likelihood")
    ax[1, 0].set_xlabel("Size of training set")
    ax[1, 0].set_ylabel("Log Likelihood")
    ax[1, 0].legend()

    ax[1, 1].plot(xs, [s.test_log_likelihood for s in nb_stats],
                  label="Naive Bayes")
    ax[1, 1].plot(xs, [s.test_log_likelihood for s in lr_stats],
                  label="Logistic Regression")
    ax[1, 1].set_title("Test Log Likelihood")
    ax[1, 1].set_xlabel("Size of training set")
    ax[1, 1].set_ylabel("Log Likelihood")
    ax[1, 1].legend()

    # set equal y limits for all graphs
    min_train_acc = min([s.train_acc for s in nb_stats + lr_stats])
    max_train_acc = max([s.train_acc for s in nb_stats + lr_stats])
    min_test_acc = min([s.test_acc for s in nb_stats + lr_stats])
    max_test_acc = max([s.test_acc for s in nb_stats + lr_stats])

    min_train_ll = min([s.train_log_likelihood for s in nb_stats + lr_stats])
    min_test_ll = min([s.test_log_likelihood for s in nb_stats + lr_stats])

    min_acc = min(min_train_acc, min_test_acc)
    max_acc = max(max_train_acc, max_test_acc)
    min_ll = min(min_train_ll, min_test_ll)
    max_ll = 0

    ax[0, 0].set_ylim(min_acc, max_acc)
    ax[0, 1].set_ylim(min_acc, max_acc)
    ax[1, 0].set_ylim(min_ll, max_ll)
    ax[1, 1].set_ylim(min_ll, max_ll)

    plt.tight_layout()
    plt.show()


data = pd.read_fwf(DATA_FILE, header=None)

benign = data[data[FEATURES] == BENIGN]
malignant = data[data[FEATURES] == MALIGNANT]

train_ds = pd.concat([benign[:int(len(benign) * TRAIN_FRAC)],
                     malignant[:int(len(malignant) * TRAIN_FRAC)]])
test_ds = pd.concat([benign[int(len(benign) * TRAIN_FRAC):],
                    malignant[int(len(malignant) * TRAIN_FRAC):]])


# shuffle the data
train_ds = train_ds.sample(frac=1).reset_index(drop=True)
test_ds = test_ds.sample(frac=1).reset_index(drop=True)

#test_on_extreme_dataset(benign, malignant)
#test_on_extreme_datasets(train_ds, test_ds)
#display_naive_bayes_info(train_ds, test_ds)
#main_comparison(train_ds, test_ds, iterations=20)
