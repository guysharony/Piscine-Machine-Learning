import pickle
import pandas as pd
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression

from z_score import zscore
from data_spliter import data_spliter
from polynomial_model import add_polynomial_features

if __name__ == "__main__":
    dataset = pd.read_csv("space_avocado.csv")

    x = dataset[['weight', 'prod_distance', 'time_delivery']].to_numpy().reshape(-1, 3)
    y = dataset[['target']].to_numpy().reshape(-1, 1)

    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)

    with open("models.pickle", "rb") as f:
        thetas = pickle.load(f)

    mse_scores = []
    y_predictions = []
    for i, theta in enumerate(thetas):
        degree = i + 1
        features = zscore(add_polynomial_features(x_test, degree))

        if features is None:
            raise AssertionError("Features not valide.")

        model = MyLinearRegression(thetas=theta)

        y_prediction = model.predict_(features)
        mse_score = model.mse_(y_test, y_prediction)

        print(f"Degree: {degree}, MSE Score: {mse_score}")

        mse_scores.append(mse_score)
        y_predictions.append(y_prediction)

    best_model = mse_scores.index(min(mse_scores))
    y_prediction = y_predictions[best_model]
    print(f"[{best_model + 1} degree] is the best model.")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        x_test[:, 0], x_test[:, 1], y_test, c="r", marker="o", label="True Price"
    )
    ax.scatter(
        x_test[:, 0], x_test[:, 1], y_prediction, c="b", marker="^", label="Predicted Price"
    )
    ax.set_xlabel("Weight")
    ax.set_ylabel("Prod Distance")
    ax.set_zlabel("Price")
    ax.legend()
    plt.show()