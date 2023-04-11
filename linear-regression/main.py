from regression.predictor import *
from regression.dataset import *


def print_mse():
    predictor = Predictor.load("predictor.model")
    ds = load_data("dane.data")
    ys = predictor(ds.xs)
    mse = (ys - ds.ys).T @ (ys - ds.ys) / len(ds)
    print("MSE:", mse)


def predict_ys():
    predictor = Predictor.load("predictor.model")
    ds = load_data("dane.data")
    ys = predictor(ds.xs)
    for y in ys:
        print(y)


if __name__ == "__main__":
    predict_ys()
    print_mse()
