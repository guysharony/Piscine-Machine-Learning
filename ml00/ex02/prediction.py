import numpy as np

def simple_predict(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exception.
    """
    if x.size == 0 or theta.size == 0:
        return None

    if x.ndim != 1 or theta.ndim != 1:
        return None

    return np.array([float(theta[0] + theta[1] * x_i) for x_i in x])

if __name__ == "__main__":
    x = np.arange(1, 6)

    # Example 1:
    theta1 = np.array([5, 0])
    print(f"<- {simple_predict(x, theta1)}")
    print(f"-> {np.array([5., 5., 5., 5., 5.])}")
    print()

    # Example 2:
    theta2 = np.array([0, 1])
    # Output:
    print(f"<- {simple_predict(x, theta2)}")
    print(f"-> {np.array([1., 2., 3., 4., 5.])}")
    print()

    # Example 3:
    theta3 = np.array([5, 3])
    # Output:
    print(f"<- {simple_predict(x, theta3)}")
    print(f"-> {np.array([8., 11., 14., 17., 20.])}")
    print()

    # Example 4:
    theta4 = np.array([-3, 1])
    # Output:
    print(f"<- {simple_predict(x, theta4)}")
    print(f"-> {np.array([-2., -1., 0., 1., 2.])}")