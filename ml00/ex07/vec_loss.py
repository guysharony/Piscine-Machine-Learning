import numpy as np

def loss_(y, y_hat):
    """Computes the half mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        The half mean squared error of the two vectors as a float.
        None if y or y_hat are empty numpy.array.
        None if y and y_hat does not share the same dimensions.
    Raises:
        This function should not raise any Exceptions.
    """
    if y.__class__ != np.ndarray or y_hat.__class__ != np.ndarray:
        return None

    if y.size != y_hat.size:
        return None

    if y.size == 0 or y_hat.size == 0:
        return None

    return (np.dot((y_hat - y).T, y_hat - y) * (1 / (2 * y.size))).item()

if __name__ == "__main__":
    X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
    Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])

    # Example 1:
    print(f"<- {loss_(X, Y)}")
    print("-> 2.142857142857143")
    print()

    # Example 2:
    print(f"<- {loss_(X, X)}")
    print("-> 0.0")