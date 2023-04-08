import numpy as np

from dataset import *
from loss import *
from base_functions import *
from gradient_descent import *
from plots import *
from predictor import *


ds = load_data("dane.data")
ds.shuffle()
normalization = normalize(ds)


projs = projections(7) + [Mean()]
polys = polynomials(4)

base_polys = [Composition(proj, poly) for proj in projs for poly in polys]
base_polys += [Mul(f, g) for f in base_polys for g in base_polys]

base_functions = [Const(1)] + base_polys  # + base_trigs + base_misc

theta = np.random.default_rng().random(len(base_functions))

print("Number of base functions: ", len(base_functions))
xs = np.array([[f(x) for f in base_functions] for x in ds.xs])
ys = ds.ys

based_ds = Dataset(xs, ys)

train_ds, val_ds, test_ds = split(based_ds, 0.2, 0.1)
loss = MSE()

model = make_model(loss, 0.0001, 1000, 50)

theta, history = model(train_ds, val_ds)

print("Train loss:", loss(theta, train_ds))
print("Val loss:", loss(theta, val_ds))
print("Test loss:", loss(theta, test_ds))

plot_train_history(history)
plot_error(ds.xs, xs @ theta - ys)

predictor = Predictor(normalization, theta, base_functions)
export_predictor(predictor, "predictor.model")
