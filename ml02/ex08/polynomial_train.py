import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression
from polynomial_model import add_polynomial_features

if __name__ == "__main__":
    data = pd.read_csv("are_blue_pills_magics.csv")
    x = data["Micrograms"].values.reshape(-1, 1)
    y = data["Score"].values.reshape(-1, 1)

    mse_scores = []
    models = []

    degrees = range(1, 7)
    for degree in degrees:
        features = add_polynomial_features(x, degree)

        if degree == 1:
            thetas = np.array([[-29],[ 115]]).reshape(-1,1)
        elif degree == 2:
            thetas = np.array([[-19],[ 160],[ -79]]).reshape(-1,1)
        elif degree == 3:
            thetas = np.array([[-20],[ 159],[ -80],[ 10]]).reshape(-1,1)
        elif degree == 4:
            thetas = np.array([[-20],[ 160],[ -80],[ 10],[ -1]]).reshape(-1,1)
        elif degree == 5:
            thetas = np.array([[1140],[ -1850],[ 1110],[ -305],[ 40],[ -2]]).reshape(-1,1)
        elif degree == 6:
            thetas = np.array([[9110],[ -18015],[ 13400],[ -4935],[ 966],[ -96.4],[ 3.86]]).reshape(-1,1)
        else:
            raise AssertionError(f"Degree but be between 1 and 6 but got {degree}.")

        # The bigger the degree the bigger is the complexity
        # so it's importent to decrease the learning rate
        learning_rate = 1 / (10000 ** (degree + 1))

        model = MyLinearRegression(
            thetas=thetas,
            alpha=learning_rate,
            max_iter=10000
        )

        model.fit_(features, y)
        y_pred = model.predict_(features)
        mse_score = model.mse_(y, y_pred)

        models.append(model)
        mse_scores.append(mse_score)

        print(f"Degree: {degree}, Mse score: {mse_score}")

    # Plotting MSE Scores
    plt.bar(degrees, mse_scores)
    plt.xlabel("Polynomial degree")
    plt.ylabel("MSE Score")
    plt.title("MSE score of the models in function of the polynomial degree.")
    plt.show()

    # Plotting models
    plt.scatter(x, y, label="Data points", color="blue")

    x_range = np.linspace(x.min(), x.max(), 10000).reshape(-1, 1)
    for degree, model in zip(degrees, models):
        # Plotting data point
        features_range = add_polynomial_features(x_range, degree)
        y_prediction_range = model.predict_(features_range)
        plt.plot(x_range, y_prediction_range, label=f"Degree {degree}")

        # Scattering models
        features_points = add_polynomial_features(x, degree)
        y_prediction_points = model.predict_(features_points)
        plt.scatter(x, y_prediction_points, marker="x")

    plt.xlabel("Micrograms")
    plt.ylabel("Scores")
    plt.show()