import numpy as np

def add_intercept(x):
    """Adds a column of 1's to the non-empty numpy.array x.
    Args:
        x: has to be a numpy.array of dimension m * n.
    Returns:
        X, a numpy.array of dimension m * (n + 1).
        None if x is not a numpy.array.
        None if x is an empty numpy.array.
    Raises:
        This function should not raise any Exception.
    """
    if x.__class__ != np.ndarray or x.size == 0:
        return None

    return np.c_[np.ones(x.shape[0]), x]


def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
        y_hat as a numpy.array, a vector of dimension m * 1.
        None if x and/or theta are not numpy.array.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exceptions.
    """
    if x.__class__ != np.ndarray or theta.__class__ != np.ndarray:
        return None

    if x.size == 0 or theta.size == 0:
        return None

    if x.ndim != 1 or theta.shape[0] != 2 or theta.shape[1] != 1:
        return None

    return np.dot(add_intercept(x), theta)

if __name__ == "__main__":
    x = np.arange(1,6)

    # Example 1:
    theta1 = np.array([[5], [0]])
    print(f"<- {predict_(x, theta1)}")
    print(f"-> {np.array([[5.], [5.], [5.], [5.], [5.]])}")
    print()

    # Example 2:
    theta2 = np.array([[0], [1]])
    print(f"<- {predict_(x, theta2)}")
    print(f"-> {np.array([[1.], [2.], [3.], [4.], [5.]])}")
    print()

    # Example 3:
    theta3 = np.array([[5], [3]])
    print(f"<- {predict_(x, theta3)}")
    print(f"-> {np.array([[ 8.], [11.], [14.], [17.], [20.]])}")
    print()

    # Example 4:
    theta4 = np.array([[-3], [1]])
    print(f"<- {predict_(x, theta4)}")
    print(f"-> {np.array([[-2.], [-1.], [ 0.], [ 1.], [ 2.]])}")