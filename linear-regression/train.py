'''
Training concrete models for tests
'''


from regression import *

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


ds = load_data("dane.data")
ds.shuffle()
normalization = normalize(ds)

# calculate_simplest_model()
# calculate_simple_model()
calculate_simple_model_l2(0.001)
