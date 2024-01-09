import numpy as np

def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for-loop. The three arrays must have the compatible dimensions.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector (n +1) * 1.
    Return:
        The gradient as a numpy.array, a vector of dimensions n * 1,
            containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible dimensions.
        None if x, y or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    m = x.shape[0]
    x_prime = np.hstack((np.ones((m, 1)), x))

    return (1 / m) * x_prime.T.dot(x_prime.dot(theta) - y)

def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.array, a matrix of dimension m * n:
                    (number of training examples, number of features).
        y: has to be a numpy.array, a vector of dimension m * 1:
                    (number of training examples, 1).
        theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
                    (number of features + 1, 1).
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Return:
        new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
        None if there is a matching dimension problem.
        None if x, y, theta, alpha or max_iter is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    theta_copy = theta.copy()
    for _ in range(max_iter):
        _gradient = gradient(x, y, theta_copy)
        theta_copy -= alpha * _gradient
    return theta_copy


def predict_(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of dimensions m * n.
        theta: has to be an numpy.array, a vector of dimensions (n + 1) * 1.
    Return:
        y_hat as a numpy.array, a vector of dimensions m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
        None if x or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if x.__class__ != np.ndarray or theta.__class__ != np.ndarray:
        return None

    if x.size == 0 or theta.size == 0:
        return None

    if theta.shape[0] - 1 != x.shape[1] or theta.shape[1] != 1:
        return None

    return np.dot(np.c_[np.ones(x.shape[0]), x], theta)


if __name__ == "__main__":
    x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
    y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])

    theta = np.array([[42.], [1.], [1.], [1.]])
    # Example 0:
    theta2 = fit_(x, y, theta, alpha = 0.0005, max_iter=42000)
    print(f"<- {theta2}")
    print(f"-> {np.array([[41.99888822],[0.97792316], [0.77923161], [-1.20768386]])}")
    print()

    # Example 1:
    print(f"<- {predict_(x, theta2)}")
    print(f"-> {np.array([[19.59925884], [-2.80037055], [-25.19999994], [-47.59962933]])}")