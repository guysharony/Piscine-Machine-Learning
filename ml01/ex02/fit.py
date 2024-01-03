import numpy as np


def simple_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for loop.
        The three arrays must have compatible shapes.
    Args:
        x: has to be a numpy.array, a matrix of shape m * 1.
        y: has to be a numpy.array, a vector of shape m * 1.
        theta: has to be a numpy.array, a 2 * 1 vector.
    Return:
        The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
        None if x, y, or theta is an empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
    Raises:
        This function should not raise any Exception.
    """
    if y.__class__ != np.ndarray or x.__class__ != np.ndarray or theta.__class__ != np.ndarray:
        return None

    if y.size == 0 or x.size == 0 or theta.size == 0:
        return None

    m = x.shape[0]
    x_prime = np.c_[np.ones(m), x]
    diff = np.dot(x_prime, theta) - y
    gradient = np.dot(x_prime.T, diff) / m
    return gradient


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Returns:
        new_theta: numpy.ndarray, a vector of dimension 2 * 1.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exception.
    """
    if y.__class__ != np.ndarray or x.__class__ != np.ndarray or theta.__class__ != np.ndarray:
        return None

    if y.size == 0 or x.size == 0 or theta.size == 0:
        return None

    updated_theta = theta.astype(float).copy()
    for _ in range(max_iter):
        gradient = simple_gradient(x, y, updated_theta)
        updated_theta -= alpha * gradient
    return updated_theta

def predict(x, theta):
    return np.dot(np.c_[np.ones(x.shape[0]), x], theta)


if __name__ == "__main__":
    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
    theta = np.array([1, 1]).reshape((-1, 1))

    # Example 0:
    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
    print(f"<- {theta1}")
    print(f"-> {np.array([[1.40709365], [1.1150909 ]])}")
    print()

    # Example 1:
    print(f"<- {predict(x, theta1)}")
    print(f"-> {np.array([[15.3408728 ], [25.38243697], [36.59126492], [55.95130097], [65.53471499]])}")