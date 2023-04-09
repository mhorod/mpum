import numpy as np

from dataset import *
from loss import *
from gradient_descent import *
from plots import *
from predictor import *
from models import *

ds = load_data("dane.data")
ds.shuffle()
normalization = normalize(ds)

model = make_simplest_model()

prepared_ds = model.prepare_dataset(ds)

train_ds, val_ds, test_ds = split(prepared_ds, 0.2, 0.1)

params = DescentParams(
    model,
    train_ds,
    val_ds,
    batch_size=50,
    epochs=1000,
    learning_rate=0.0001,
)

history = continuous_gradient_descent(params)

print("Train loss:", model.evaluate(train_ds))
print("Val loss:", model.evaluate(val_ds))
print("Test loss:", model.evaluate(test_ds))
'''
marked_percentages = [0.01, 0.02, 0.03, 0.125, 0.625, 1]
percentages = sorted(set(
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] + marked_percentages))

marked_percentages = [percentages.index(p) for p in marked_percentages]
curve = make_average_learning_curve(train_ds, test_ds, percentages, 10, params)
plot_learning_curve(percentages, curve, marked_percentages)
'''

plot_train_history(history)
plot_by_x(ds.xs, prepared_ds.xs @ model.theta - prepared_ds.ys)


#predictor = Predictor(normalization, theta, base_functions)
#export_predictor(predictor, "predictor.model")
