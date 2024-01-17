import numpy as np

def reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.ndarray,
        with two for-loop. The three arrays must have compatible shapes.
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
        lambda_: has to be a float.
    Return:
        A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
        None if y, x or theta or lambda_ is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if x.__class__ != np.ndarray or y.__class__ != np.ndarray or theta.__class__ != np.ndarray:
        return None

    if x.size == 0 or y.size == 0 or theta.size == 0:
        return None

    if y.shape[0] != x.shape[0]:
        return None

    if x.shape[1] + 1 != theta.shape[0]:
        return None

    m, n = x.shape
    x_prime = np.hstack((np.ones((x.shape[0], 1)), x))
    gradient = np.zeros((n + 1, 1))

    for j in range(n + 1):
        for i in range(m):
            gradient[j] += (np.dot(x_prime[i], theta) - y[i]) * x_prime[i, j]
        if j != 0:
            gradient[j] += lambda_ * theta[j]

    return gradient / m


def vec_reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.ndarray,
        without any for-loop. The three arrays must have compatible shapes.
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
        lambda_: has to be a float.
    Return:
        A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
        None if y, x or theta or lambda_ is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if x.__class__ != np.ndarray or y.__class__ != np.ndarray or theta.__class__ != np.ndarray:
        return None

    if x.size == 0 or y.size == 0 or theta.size == 0:
        return None
    
    if y.shape[0] != x.shape[0]:
        return None

    if x.shape[1] + 1 != theta.shape[0]:
        return None

    m = x.shape[0]
    x_prime = np.hstack((np.ones((m, 1)), x))
    theta_prime = theta.copy()
    theta_prime[0] = 0

    gradient = (1 / m) * (
        np.dot(x_prime.T, (np.dot(x_prime, theta) - y)) + (lambda_ * theta_prime)
    )

    return gradient


if __name__ == "__main__":
    x = np.array([
    [ -6, -7, -9], [ 13, -2, 14], [ -7, 14, -1], [-8, -4, 6], [-5, -9, 6], [ 1, -5, 11], [9,-11, 8]])
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    theta = np.array([[7.01], [3], [10.5], [-6]])
    
    # Example 1.1:
    print(f"<- {reg_linear_grad(y, x, theta, 1)}")
    output = np.array([[ -60.99 ],
                    [-195.64714286],
                    [ 863.46571429],
                    [-644.52142857]])
    print(f"-> {output}")
    print()

    # Example 1.2:
    print(f"<- {vec_reg_linear_grad(y, x, theta, 1)}")
    output = np.array([[ -60.99 ],
                    [-195.64714286],
                    [ 863.46571429],
                    [-644.52142857]])
    print(f"-> {output}")
    print()

    # Example 2.1:
    print(f"<- {reg_linear_grad(y, x, theta, 0.5)}")
    output = np.array([[ -60.99 ],
                    [-195.86142857],
                    [ 862.71571429],
                    [-644.09285714]])
    print(f"-> {output}")
    print()

    # Example 2.2:
    print(f"<- {vec_reg_linear_grad(y, x, theta, 0.5)}")
    output = np.array([[ -60.99 ],
                    [-195.86142857],
                    [ 862.71571429],
                    [-644.09285714]])
    print(f"-> {output}")
    print()

    # Example 3.1:
    print(f"<- {reg_linear_grad(y, x, theta, 0.0)}")
    output = np.array([[ -60.99 ],
                    [-196.07571429],
                    [ 861.96571429],
                    [-643.66428571]])
    print(f"-> {output}")
    print()

    # Example 3.2:
    print(f"<- {vec_reg_linear_grad(y, x, theta, 0.0)}")
    output = np.array([[ -60.99 ],
                    [-196.07571429],
                    [ 861.96571429],
                    [-643.66428571]])
    print(f"-> {output}")