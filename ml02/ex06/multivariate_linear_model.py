import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression as MyLR



if __name__ == "__main__":
    data = pd.read_csv("spacecraft_data.csv")
    y = np.array(data[['Sell_price']])

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
    plt.scatter(x_age, y_age_pred, label="Predicted sell price", zorder=3)
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
    plt.scatter(x_thrust, y_thrust_pred, label="Predicted sell price", zorder=3)
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