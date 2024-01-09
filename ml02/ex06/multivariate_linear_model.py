import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression as MyLR



if __name__ == "__main__":
    data = pd.read_csv("spacecraft_data.csv")
    y = np.array(data[['Sell_price']])


    ## Part One

    # Age
    x_age = np.array(data[['Age']])

    myLR_age = MyLR(
        thetas = np.array([[1000.0], [-1.0]]),
        max_iter = 400000,
        alpha = 0.0001,
    )
    myLR_age.fit_(x_age, y)
    y_age_pred = myLR_age.predict_(x_age)

    print(f"Age mse => {myLR_age.mse_(y_age_pred, y)}")

    plt.scatter(x_age, y, label="Sell price", color="blue", zorder=3)
    plt.scatter(x_age, y_age_pred, label="Predicted sell price", color="dodgerblue", zorder=3)
    plt.xlabel("x1: age (in years)")
    plt.ylabel("y: sell price (in keuros)")
    plt.title("The selling prices of spacecrafts with respect to their age.")
    plt.legend()
    plt.grid(True, zorder=0)
    plt.show()

    # Thrust
    x_thrust = np.array(data[['Thrust_power']])

    myLR_thrust = MyLR(
        thetas = np.array([[1000.0], [-1.0]]),
        max_iter = 400000,
        alpha = 0.0001,
    )
    myLR_thrust.fit_(x_thrust, y)
    y_thrust_pred = myLR_thrust.predict_(x_thrust)

    print(f"Thrust mse => {myLR_thrust.mse_(y_thrust_pred, y)}")

    plt.scatter(x_thrust, y, label="Sell price", color="green", zorder=3)
    plt.scatter(x_thrust, y_thrust_pred, label="Predicted sell price", color="lime", zorder=3)
    plt.xlabel("x2: thrust power (in 10km/s)")
    plt.ylabel("y: sell price (in keuros)")
    plt.title("The selling prices of spacecrafts with respect to the thrust power of their engines.")
    plt.legend()
    plt.grid(True, zorder=0)
    plt.show()

    # Terameters
    x_terameter = np.array(data[['Terameters']])

    myLR_terameter = MyLR(
        thetas = np.array([[1000.0], [-1.0]]),
        max_iter = 400000,
        alpha = 0.0001,
    )
    myLR_terameter.fit_(x_terameter, y)
    y_terameter_pred = myLR_terameter.predict_(x_terameter)

    print(f"Terameter mse => {myLR_terameter.mse_(y_terameter_pred, y)}")

    plt.scatter(x_terameter, y, label="Sell price", color="purple", zorder=3)
    plt.scatter(x_terameter, y_terameter_pred, label="Predicted sell price", color="pink", zorder=3)
    plt.xlabel("x3: distance totalizer value of spacecraft (in Tmeters)")
    plt.ylabel("y: sell price (in keuros)")
    plt.title("The selling prices of spacecrafts with respect to the terameters driven.")
    plt.legend()
    plt.grid(True, zorder=0)
    plt.show()


    ## Part Two
    x = np.array(data[['Age', 'Thrust_power', 'Terameters']])
    y = np.array(data[['Sell_price']])
    my_lreg = MyLR(
        thetas = np.ones((4, 1), dtype=float),
        alpha = 1e-5,
        max_iter = 600000
    )
    my_lreg.fit_(x, y)
    y_pred = my_lreg.predict_(x)

    # Age
    plt.scatter(x_age, y, label="Sell price", color="blue", zorder=3)
    plt.scatter(x_age, y_pred, label="Predicted sell price", color="dodgerblue", zorder=3)
    plt.xlabel("x1: age (in years)")
    plt.ylabel("y: sell price (in keuros)")
    plt.title("Spacecraft sell prices of and predicted sell prices with the multivariate hypothesis.")
    plt.legend()
    plt.grid(True, zorder=1)
    plt.show()

    # Trust
    plt.scatter(x_thrust, y, label="Sell price", color="green", zorder=3)
    plt.scatter(x_thrust, y_pred, label="Predicted sell price", color="lime", zorder=3)
    plt.xlabel("x1: thrust power (in 10km/s)")
    plt.ylabel("y: sell price (in keuros)")
    plt.title("Spacecraft sell prices predicted sell prices with the multivariate hypothesis.")
    plt.legend()
    plt.grid(True, zorder=1)
    plt.show()

    # Trust
    plt.scatter(x_terameter, y, label="Sell price", color="purple", zorder=3)
    plt.scatter(x_terameter, y_pred, label="Predicted sell price", color="pink", zorder=3)
    plt.xlabel("x1: distance totalizer value of spacecraft (in Tmeters)")
    plt.ylabel("y: sell price (in keuros)")
    plt.title("Spacecraft sell prices predicted sell prices with the multivariate hypothesis.")
    plt.legend()
    plt.grid(True, zorder=1)
    plt.show()