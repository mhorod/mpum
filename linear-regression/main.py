import numpy as np

from dataset import *
from loss import *
from base_functions import *
from gradient_descent import *
from plots import *


def make_learning_curve(loss, train_ds, learning_rate, epochs, batch_size, percentages):
    theta = np.random.default_rng().random(len(train_ds.xs[0]))

    losses = []
    for p in percentages:
        ds = train_ds[:int(len(train_ds) * p)]
        theta, history = continuous_gradient_descent(
            loss, theta,
            ds,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size
        )
        losses.append(history['loss'][-1])
    return losses


ds = load_data("dane.data")
ds.shuffle()
standarization = standarize(ds)


projs = projections(7)
polys = polynomials(4)

base_functions = [compose(proj, poly) for proj in projs for poly in polys]
base_functions += [mul(f, g)
                   for f in base_functions for g in base_functions]
base_functions = [const(1)] + base_functions

theta = np.random.default_rng().random(len(base_functions))

print(len(base_functions))
xs = np.array([[f(x) for f in base_functions] for x in ds.xs])
ys = ds.ys

based_ds = Dataset(xs, ys)

train_ds, val_ds, test_ds = split(based_ds, 0.2, 0.1)
loss = MSE()

analytic = np.linalg.pinv(xs.T @ xs) @ xs.T @ ys


print("Analytic test loss:", loss(analytic, test_ds))

'''
theta, history = continuous_gradient_descent(
    loss, theta,
    train_ds,
    learning_rate=0.0001,
    epochs=100,
    val_ds=val_ds,
    batch_size=50
)


print("Train loss:", loss(theta, train_ds))
print("Val loss:", loss(theta, val_ds))
print("Test loss:", loss(theta, test_ds))

plot_train_history(history)
plot_error(ds.xs, xs @ theta - ys)
# plot_error(ds.xs, xs @ analytic - ys)
'''

percentages = list(np.linspace(0.05, 1, 20))
marked_percentages = [0.01, 0.02, 0.03, 0.125, 0.625, 1]

percentages = sorted(set(percentages + marked_percentages))

learning_rate = 0.0001
epochs = 100
batch_size = 50
curve = make_learning_curve(
    loss, train_ds, learning_rate, epochs, batch_size, percentages)

marked_percentages = [percentages.index(p) for p in marked_percentages]
plot_learning_curve(percentages, curve, marked_percentages)
