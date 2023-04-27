from dataclasses import dataclass

import numpy as np

from common import *


@dataclass
class BayesClassifier:
    y_phi: float
    x_phis: np.ndarray


def make_naive_bayes(ds):
    benign_count = len(ds[ds[FEATURES] == BENIGN])
    malignant_count = len(ds[ds[FEATURES] == MALIGNANT])

    L = 1  # laplace smoothing factor
    x_phis = np.zeros((2, FEATURES, FEATURE_VALUES))
    y_phi = (L + malignant_count) / (L * 2 + len(ds))

    for i in range(FEATURES):
        for j in range(1, FEATURE_VALUES + 1):
            benign_cases = len(ds[(ds[i] == j) & (ds[FEATURES] == BENIGN)])
            malignant_cases = len(
                ds[(ds[i] == j) & (ds[FEATURES] == MALIGNANT)])

            x_phis[0][i][j - 1] = (L + benign_cases) / (2 * L + benign_count)
            x_phis[1][i][j - 1] = (L + malignant_cases) / \
                (2 * L + malignant_count)

    return BayesClassifier(y_phi, x_phis)


def predict_one(x, classifier):
    p_prior_malignant = classifier.y_phi

    px_benign = (1 - p_prior_malignant)
    px_malignant = p_prior_malignant

    for i in range(FEATURES):
        px_benign *= classifier.x_phis[0][i][x[i] - 1]
        px_malignant *= classifier.x_phis[1][i][x[i] - 1]

    px = px_benign + px_malignant
    p_posterior_malignant = px_malignant / px
    if p_posterior_malignant > 0.5:
        return MALIGNANT
    else:
        return BENIGN


def measure_accuracy(ds, classifier):
    correct = 0
    for row in ds.values:
        x = row[:FEATURES]
        y = row[FEATURES]
        y_hat = predict_one(x, classifier)
        if y == y_hat:
            correct += 1

    return correct / len(ds)


def log_likelihood(ds, classifier):
    l = 0
    for row in ds.values:
        p = 1
        x = row[:FEATURES]
        y = row[FEATURES]

        px_y0 = 1
        px_y1 = 1

        for feature in range(FEATURES):
            px_y0 *= classifier.x_phis[0][feature][x[feature] - 1]
            px_y1 *= classifier.x_phis[1][feature][x[feature] - 1]

        px = np.log(px_y0 * (1 - classifier.y_phi) + px_y1 * classifier.y_phi)

        if y == MALIGNANT:
            p = np.log(classifier.y_phi) - px
            for feature in range(FEATURES):
                p += np.log(classifier.x_phis[1][feature][x[feature] - 1])
        else:
            p = np.log(1 - classifier.y_phi) - px
            for feature in range(FEATURES):
                p += np.log(classifier.x_phis[0][feature][x[feature] - 1])

        l += p

    return l / len(ds)
