import numpy as np

def l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if theta.__class__ != np.ndarray:
        return None

    if theta.size == 0:
        return None
    
    theta_prime = theta.copy()
    theta_prime[0] = 0.0
    l2_reg = np.dot(theta_prime.T, theta_prime)

    return l2_reg[0][0].astype(float)

def reg_log_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a logistic regression model from two non-empty numpy.ndarray, without any for loop.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        The regularized loss as a float.
        None if y, y_hat, or theta is empty numpy.ndarray.
        None if y and y_hat do not share the same shapes.
    Raises:
        This function should not raise any Exception.
    """
    if y.__class__ != np.ndarray or y_hat.__class__ != np.ndarray or theta.__class__ != np.ndarray:
        return None

    if y.size == 0 or y_hat.size == 0 or theta.size == 0:
        return None

    if y.shape != y_hat.shape:
        return None
    
    m = y.shape[0]
    ones_vector = np.ones(y.shape)
    loss = (-1 / m) * (
        np.dot(y.T, np.log(y_hat)) + np.dot((ones_vector - y).T, np.log(ones_vector - y_hat))
    ) + (lambda_ / (2 * m)) * l2(theta)

    return loss[0][0].astype(float)

if __name__ == "__main__":
    y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
    y_hat = np.array([.9, .79, .12, .04, .89, .93, .01]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))

    # Example 1:
    print(f"<- {reg_log_loss_(y, y_hat, theta, .5)}")
    print(f"-> {0.43377043716476066}")
    print()

    # Example 2:
    print(f"<- {reg_log_loss_(y, y_hat, theta, .05)}")
    print(f"-> {0.13452043716476064}")
    print()

    # Example 3:
    print(f"<- {reg_log_loss_(y, y_hat, theta, .9)}")
    print(f"-> {0.6997704371647608}")