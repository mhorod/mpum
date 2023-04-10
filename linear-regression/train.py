from analytic import *
from dataset import *
from loss import *
from gradient_descent import *
from plots import *
from predictor import *
from models import *

import os
import shutil


def print_loss(model, train_ds, val_ds, test_ds, model_name=None):
    if model_name is not None:
        print(model_name)
    print("Train loss:", model.evaluate(train_ds))
    print("Val loss:", model.evaluate(val_ds))
    print("Test loss:", model.evaluate(test_ds))
    if model_name is not None:
        print()


def calculate_model(model, ds, descent_meta_params, descent, model_name, analytic=None):
    shutil.rmtree(f"img/{model_name}", ignore_errors=True)
    os.mkdir(f"img/{model_name}")

    prepared_ds = model.prepare_dataset(ds)
    train_ds, val_ds, test_ds = split(prepared_ds, 0.2, 0.1)

    if analytic is not None:
        theta = analytic(prepared_ds)
        analytic_model = Model(model.loss, model.base_functions, theta)
        print_loss(analytic_model, train_ds, val_ds, test_ds,
                   f"[{model_name}] Analytic")

        plot_by_x(ds.xs, prepared_ds.xs @ theta - prepared_ds.ys,
                  "green", filename=f"img/{model_name}/analytic-error.png")

    descent_params = descent_meta_params.into_descent_params(
        model, train_ds, val_ds)
    history = descent(descent_params)

    print_loss(model, train_ds, val_ds, test_ds,
               f"[{model_name}] Gradient descent")

    plot_train_history(history, f"img/{model_name}/history.png")
    plot_by_x(ds.xs, prepared_ds.xs @ model.theta - prepared_ds.ys,
              filename=f"img/{model_name}/descent-error.png")

    percentages = [0.01, 0.02, 0.03, 0.125, 0.625, 1]

    marked_percentages = list(range(len(percentages)))
    curve = make_average_learning_curve(
        train_ds, test_ds, percentages, 5, descent_params)
    plot_learning_curve(percentages, curve, marked_percentages,
                        f"img/{model_name}/learning-curve.png")


ds = load_data("dane.data")
ds.shuffle()
normalization = normalize(ds)

model = make_simplest_model()

descent_meta_params = DescentMetaParams(
    batch_size=50,
    epochs=1000,
    learning_rate=0.0001,
)

calculate_model(
    model,
    ds,
    descent_meta_params,
    continuous_gradient_descent,
    "Simplest model",
    analytic=analytic_mse
)

calculate_model(
    model,
    ds,
    descent_meta_params,
    continuous_gradient_descent_with_momentum,
    "Simplest model [momentum]",
    analytic=analytic_mse
)
