import numpy as np
import pandas as pd
from data_spliter import data_spliter
from my_logistic_regression import MyLogisticRegression


if __name__ == "__main__":
    # Extracting data from model
    x_dataset = pd.read_csv("solar_system_census.csv")
    x_dataset.drop("Unnamed: 0", axis=1, inplace=True)

    y_dataset = pd.read_csv("solar_system_census_planets.csv")
    y_dataset.drop("Unnamed: 0", axis=1, inplace=True)

    x = x_dataset.to_numpy()
    y = y_dataset["Origin"].astype(int).to_numpy().reshape(-1, 1)


    # Spliting data into training and testing data  
    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)


    # Training model
    models = []
    for i in range(4):
        y_train_class = (y_train == i).astype(int)

        model = MyLogisticRegression(
            thetas=np.zeros((x.shape[1] + 1, 1)),
            alpha=0.001,
            max_iter=1_000_000
        )
        model.fit_(x_train, y_train_class)

        models.append(model)

    y_predictions_probabilities = [model.predict_(x_test) for model in models]

    print(y_predictions_probabilities)
    print()
    print(np.hstack(y_predictions_probabilities))