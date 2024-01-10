import pickle
import numpy as np
import pandas as pd
from z_score import zscore
from data_spliter import data_spliter
from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression

if __name__ == "__main__":
    dataset = pd.read_csv("space_avocado.csv")

    x = dataset[['weight', 'prod_distance', 'time_delivery']].to_numpy().reshape(-1, 3)
    y = dataset['target'].to_numpy().reshape(-1, 1)

    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)

    mse_scores = []
    thetas = []
    for degree in range(1, 5):
        features = zscore(add_polynomial_features(x_train, degree))

        if features is None:
            raise AssertionError("Features not valide.")

        model = MyLinearRegression(
            thetas=np.zeros((features.shape[1] + 1, 1)),
            alpha=0.005,
            max_iter=1000000
        )

        model.fit_(features, y_train)
        y_pred = model.predict_(features)
        mse_score = model.mse_(y_train, y_pred)

        thetas.append(model.thetas)
        mse_scores.append(mse_score)

        print(f"Degree: {degree}, MSE Score: {mse_score}")
        print(f"Thetas: {model.thetas.flatten()}")
        print()

    with open("models.pickle", "wb") as f:
        pickle.dump(thetas, f)
