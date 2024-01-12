import argparse
import pandas as pd
from data_spliter import data_spliter
from my_logistic_regression import MyLogisticRegression
import numpy as np

class CheckZipcode(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        zipcode = int(values)
        if zipcode < 0 or zipcode > 3:
            raise argparse.ArgumentTypeError(f"-zipcode must be between 0 and 3 included.")
        setattr(namespace, self.dest, zipcode)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="Zipcode of your favourite planet."
        )
        parser.add_argument(
            "-zipcode",
            type=int,
            action=CheckZipcode,
            help='Value between 0 and 3 included.'
        )

        args = parser.parse_args()
        zipcode = args.zipcode
        if zipcode is None:
            parser.print_usage()
            exit(1)


        # Extracting data from model
        x_dataset = pd.read_csv("solar_system_census.csv")
        x_dataset.drop("Unnamed: 0", axis=1, inplace=True)

        y_dataset = pd.read_csv("solar_system_census_planets.csv")
        y_dataset.drop("Unnamed: 0", axis=1, inplace=True)

        x = x_dataset.to_numpy()
        y = (y_dataset["Origin"] == zipcode).astype(int).to_numpy().reshape(-1, 1)


        # Spliting data into training and testing data  
        x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)


        # Training model
        myLR = MyLogisticRegression(
            thetas=np.zeros((x.shape[1] + 1, 1)),
            alpha=0.001,
            max_iter=1_000_000
        )
        myLR.fit_(x_train, y_train)


        # Prediction
        y_prediction = myLR.predict_(x_test)

        # Accuracy
        assert y_prediction is not None, "Prediction failed."

        print(f'Accuracy: {np.mean(y_test == (y_prediction >= 0.5)) * 100}%')

    except Exception as err:
        print(f'error: {err}')