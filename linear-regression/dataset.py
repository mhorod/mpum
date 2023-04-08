from dataclasses import dataclass

import numpy as np


@dataclass
class Dataset:
    xs: np.ndarray
    ys: np.ndarray

    def __getitem__(self, i):
        return Dataset(self.xs[i], self.ys[i])

    def __len__(self):
        return len(self.xs)

    def __iter__(self):
        return zip(self.xs, self.ys)

    def shuffle(self):
        p = np.random.permutation(len(self.xs))
        self.xs = self.xs[p]
        self.ys = self.ys[p]


def load_data(path):
    with open(path, "r") as f:
        data = f.read().split("\n")
        data = [line.split("\t") for line in data if line]
        data = np.array([
            [float(x) for x in line]
            for line in data
        ])
        xs = data[:, :7]
        ys = data[:, 7]
        return Dataset(xs, ys)


def standardized(xs):
    mean = np.mean(xs, axis=0)
    std = np.std(xs, axis=0)
    return (xs - mean) / std, mean, std


def standarize(ds):
    ds.xs, xs_mean, xs_std = standardized(ds.xs)
    ds.ys, ys_mean, ys_std = standardized(ds.ys)
    return (xs_mean, xs_std), (ys_mean, ys_std)


def split(ds, val_fraction, test_fraction):
    i = int(len(ds) * val_fraction)
    j = i + int(len(ds) * test_fraction)

    val_ds = ds[:i]
    test_ds = ds[i:j]
    train_ds = ds[j:]
    return train_ds, val_ds, test_ds
