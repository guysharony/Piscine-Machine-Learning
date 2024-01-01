import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt


def mse_(y, y_hat):
    """
    Description:
        Calculate the MSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        mse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    if y.__class__ != np.ndarray or y_hat.__class__ != np.ndarray:
        return None

    if y.size != y_hat.size:
        return None

    if y.size == 0 or y_hat.size == 0:
        return None

    return (np.dot((y_hat - y).T, y_hat - y) / y.size).item()


def rmse_(y, y_hat):
    """
    Description:
        Calculate the RMSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        rmse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    mse = mse_(y, y_hat)

    return None if mse is None else sqrt(mse)


def mae_(y, y_hat):
    """
    Description:
        Calculate the MAE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        mae: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    if y.__class__ != np.ndarray or y_hat.__class__ != np.ndarray:
        return None

    if y.size != y_hat.size:
        return None

    if y.size == 0 or y_hat.size == 0:
        return None

    return np.sum(np.absolute(y_hat - y)) / y.size


def r2score_(y, y_hat):
    """
    Description:
        Calculate the R2score between the predicted output and the output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        r2score: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """


if __name__ == "__main__":
    # Example 1:
    x = np.array([0, 15, -9, 7, 12, 3, -21])
    y = np.array([2, 14, -13, 5, 12, 4, -19])
    # Mean squared error
    ## your implementation
    print(f"<- {mse_(x,y)}")
    print("-> 4.285714285714286")

    ## sklearn implementation
    print(f"<- {mean_squared_error(x,y)}")
    print("-> 4.285714285714286")

    # Root mean squared error
    ## your implementation
    print(f"<- {rmse_(x,y)}")
    print("-> 2.0701966780270626")

    ## sklearn implementation not available: take the square root of MSE
    print(f"<- {sqrt(mean_squared_error(x,y))}")
    print("-> 2.0701966780270626")

    # Mean absolute error
    ## your implementation
    print(f"<- {mae_(x,y)}")
    print("-> 1.7142857142857142")

    ## sklearn implementation
    print(f"<- {mean_absolute_error(x,y)}")
    print("-> 1.7142857142857142")

    # R2-score
    ## your implementation
    print(f"<- {r2score_(x,y)}")
    print("-> 0.9681721733858745")

    ## sklearn implementation
    print(f"<- {r2_score(x,y)}")
    print("-> 0.9681721733858745")