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


if __name__ == "__main__":
    x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
    y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))

    # Example 0:
    theta1 = np.array([2, 0.7]).reshape((-1, 1))
    print(f"<- {simple_gradient(x, y, theta1)}")
    print(f"-> {np.array([[-19.0342574], [-586.66875564]])}")
    print()

    # Example 1:
    theta2 = np.array([1, -0.4]).reshape((-1, 1))
    print(f"<- {simple_gradient(x, y, theta2)}")
    print(f"-> {np.array([[-57.86823748], [-2230.12297889]])}")
