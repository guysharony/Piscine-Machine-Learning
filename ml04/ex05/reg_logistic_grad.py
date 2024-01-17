import numpy as np

def reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.ndarray, with two for-loops. The three array
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
    Raises:
        This function should not raise any Exception.
    """


def vec_reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.ndarray, without any for-loop. The three arr
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of shape m * n.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
    Raises:
        This function should not raise any Exception.
    """


if __name__ == "__main__":
    x = np.array([[0, 2, 3, 4],
                    [2, 4, 5, 5],
                    [1, 3, 2, 7]])
    y = np.array([[0], [1], [1]])
    theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

    # Example 1.1:
    print(f"<- {reg_logistic_grad(y, x, theta, 1)}")
    output = np.array([[-0.55711039],
                    [-1.40334809],
                    [-1.91756886],
                    [-2.56737958],
                    [-3.03924017]])
    print(f"-> {output}")
    print()

    # Example 1.2:
    print(f"<- {vec_reg_logistic_grad(y, x, theta, 1)}")
    output = np.array([[-0.55711039],
                    [-1.40334809],
                    [-1.91756886],
                    [-2.56737958],
                    [-3.03924017]])
    print(f"-> {output}")
    print()

    # Example 2.1:
    print(f"<- {reg_logistic_grad(y, x, theta, 0.5)}")
    output = np.array([[-0.55711039],
                    [-1.15334809],
                    [-1.96756886],
                    [-2.33404624],
                    [-3.15590684]])
    print(f"-> {output}")
    print()

    # Example 2.2:
    print(f"<- {vec_reg_logistic_grad(y, x, theta, 0.5)}")
    output = np.array([[-0.55711039],
                    [-1.15334809],
                    [-1.96756886],
                    [-2.33404624],
                    [-3.15590684]])
    print(f"-> {output}")
    print()

    # Example 3.1:
    print(f"<- {reg_logistic_grad(y, x, theta, 0.0)}")
    output = np.array([[-0.55711039],
                    [-0.90334809],
                    [-2.01756886],
                    [-2.10071291],
                    [-3.27257351]])
    print(f"-> {output}")
    print()

    # Example 3.2:
    print(f"<- {vec_reg_logistic_grad(y, x, theta, 0.0)}")
    output = np.array([[-0.55711039],
                    [-0.90334809],
                    [-2.01756886],
                    [-2.10071291],
                    [-3.27257351]])
    print(f"-> {output}")