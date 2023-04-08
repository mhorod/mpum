from predictor import *
from dataset import *


def main():
    predictor = load_predictor("predictor.model")
    ds = load_data("dane.data")
    ys = predictor(ds.xs)
    mse = (ys - ds.ys).T @ (ys - ds.ys) / len(ds)
    print("MSE:", mse)


if __name__ == "__main__":
    main()
