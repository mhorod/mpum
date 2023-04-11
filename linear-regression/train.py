'''
Training concrete models for tests
'''

from regression import *

import os
import shutil

ds = load_data("dane.data")
ds.shuffle()
normalization = normalize(ds)


def print_loss(model, train_ds, val_ds, test_ds, model_name=None):
    if model_name is not None:
        print(model_name)
    print("Train loss:", model.evaluate(train_ds))
    print("Val loss:", model.evaluate(val_ds))
    print("Test loss:", model.evaluate(test_ds))
    if model_name is not None:
        print()


def calculate_model(model, ds, descent_meta_params, descent, model_name, analytic=None, learning_curve=True):
    shutil.rmtree(f"img/{model_name}", ignore_errors=True)
    os.mkdir(f"img/{model_name}")

    print(f"Model: {model_name}")
    print(f"Features: {model.features()}")
    print(f"Preparing dataset...")
    prepared_ds = model.prepare_dataset(ds)
    train_ds, val_ds, test_ds = split(prepared_ds, 0.2, 0.1)

    if analytic is not None:
        print("Calculating analytic solution...")
        theta = analytic(prepared_ds)
        analytic_model = Model(model.loss, model.base_functions, theta)
        print_loss(analytic_model, train_ds, val_ds, test_ds,
                   f"[{model_name}] Analytic")

        plot_by_x(ds.xs, prepared_ds.xs @ theta - prepared_ds.ys,
                  "green", filename=f"img/{model_name}/analytic-error.png")

    print("Calculating gradient descent...")
    descent_params = descent_meta_params.into_descent_params(
        model, train_ds, val_ds)
    history = descent(descent_params)

    print_loss(model, train_ds, val_ds, test_ds,
               f"[{model_name}] Gradient descent")

    plot_train_history(history, f"img/{model_name}/history.png")
    plot_by_x(ds.xs, prepared_ds.xs @ model.theta - prepared_ds.ys,
              filename=f"img/{model_name}/descent-error.png")

    if learning_curve:
        percentages = [0.01, 0.02, 0.03, 0.125, 0.625, 1]
        marked_percentages = list(range(len(percentages)))
        curve = make_average_learning_curve(
            train_ds, test_ds, percentages, 5, descent_params)
        plot_learning_curve(percentages, curve, marked_percentages,
                            f"img/{model_name}/learning-curve.png")


def calculate_simplest_model():
    simplest_model = make_simplest_model()
    descent_meta_params = DescentMetaParams(
        batch_size=50,
        epochs=1000,
        learning_rate=0.0001,
    )

    calculate_model(
        simplest_model,
        ds,
        descent_meta_params,
        continuous_gradient_descent,
        "simplest-model",
        analytic=analytic_mse
    )

    calculate_model(
        simplest_model,
        ds,
        descent_meta_params,
        continuous_gradient_descent_with_momentum,
        "simplest-model-momentum",
        analytic=analytic_mse
    )


def calculate_simple_model():
    simple_model = make_simple_model()
    descent_meta_params = DescentMetaParams(
        batch_size=50,
        epochs=1000,
        learning_rate=0.00001,
    )

    calculate_model(
        simple_model,
        ds,
        descent_meta_params,
        continuous_gradient_descent,
        "simple-model",
        analytic=analytic_mse
    )

    calculate_model(
        simple_model,
        ds,
        descent_meta_params,
        continuous_gradient_descent_with_momentum,
        "simple-model-momentum",
        analytic=analytic_mse
    )


def calculate_simple_model_l2(l2):
    simple_model_l2 = make_simple_model_l2(l2)
    descent_meta_params = DescentMetaParams(
        batch_size=50,
        epochs=1000,
        learning_rate=0.00001,
    )

    calculate_model(
        simple_model_l2,
        ds,
        descent_meta_params,
        continuous_gradient_descent,
        "simple-model-l2",
        analytic=lambda ds: analytic_l2(ds, l2)
    )

    calculate_model(
        simple_model_l2,
        ds,
        descent_meta_params,
        continuous_gradient_descent_with_momentum,
        "simple-model-l2-momentum",
        analytic=lambda ds: analytic_l2(ds, l2)
    )


def calculate_simple_model_l1(l1):
    simple_model_l1 = make_simple_model_l1(l1)
    descent_meta_params = DescentMetaParams(
        batch_size=50,
        epochs=1000,
        learning_rate=0.00001,
    )

    calculate_model(
        simple_model_l1,
        ds,
        descent_meta_params,
        coordinate_gradient_descent,
        "simple-model-l1",
    )


def calculate_simple_model_elastic_net(l1, l2):
    simple_model_elastic_net = make_simple_model_elastic_net(l1, l2)
    descent_meta_params = DescentMetaParams(
        batch_size=50,
        epochs=1000,
        learning_rate=0.00001,
    )

    calculate_model(
        simple_model_elastic_net,
        ds,
        descent_meta_params,
        coordinate_gradient_descent,
        "simple-model-elastic-net",
    )


def export_best_predictor():
    simple = make_simple_model()
    prepared_ds = simple.prepare_dataset(ds)
    simple_analytic_theta = analytic_mse(prepared_ds)
    pred = Predictor(normalization, simple_analytic_theta,
                     simple.base_functions)
    pred.export("predictor.model")


# calculate_simplest_model()
# calculate_simple_model()
# calculate_simple_model_l2(10 ** -4)
# calculate_simple_model_l1(10 ** -4)
# calculate_simple_model_elastic_net(10 ** -4, 10 ** -4)
# export_best_predictor()
