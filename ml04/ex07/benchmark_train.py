import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from z_score import zscore
from data_spliter import data_spliter
from polynomial_model import add_polynomial_features
from ridge import MyRidge
import copy

if __name__ == "__main__":
    dataset = pd.read_csv("space_avocado.csv")

    x = dataset[['weight', 'prod_distance', 'time_delivery']].to_numpy().reshape(-1, 3)
    y = dataset['target'].to_numpy().reshape(-1, 1)

    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)

    models = []
    for degree in range(1, 5):
        for lambda_ in np.arange(0, 1.2, step=0.2):
            features = zscore(add_polynomial_features(x_train, degree))

            if features is None:
                raise AssertionError("Features not valide.")

            model = MyRidge(
                thetas=np.ones(shape=(3 * degree + 1, 1)),
                alpha=0.005,
                max_iter=100000,
                lambda_=lambda_,
                degree=degree,
            )

            model.fit_(features, y_train)
            y_pred = model.predict_(features)

            model.loss = model.loss_(y_train, y_pred)
            models.append(copy.deepcopy(model))

            print(f"Degree: {model.degree}, Loss Score: {model.loss}")
            print(f"Thetas: {model.thetas.flatten()}")
            print()

    with open("models.pickle", "wb") as f:
        pickle.dump(models, f)

    plt.title("Loss vs Degree and Lambda")
    xticks = [f"Degree {model.degree} λ/{model.lambda_:.1f}" for model in models]
    plt.xticks(np.arange(len(xticks)), xticks, rotation=90)
    plt.plot(np.arange(len(models)), [model.loss for model in models])
    plt.ylabel("Loss")
    plt.xlabel("Degree and Lambda")
    plt.show()