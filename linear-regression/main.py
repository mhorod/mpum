from regression.predictor import *
from regression.dataset import *


def main():
    predictor = Predictor.load("predictor.model")
    ds = load_data("dane.data")
    ys = predictor(ds.xs)
    mse = (ys - ds.ys).T @ (ys - ds.ys) / len(ds)
    print("MSE:", mse)


if __name__ == "__main__":
    main()
