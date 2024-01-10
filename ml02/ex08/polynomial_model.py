import numpy as np

def add_polynomial_features(x, power):
    """Add polynomial features to vector x by raising its values up to the power given in argument.
    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        power: has to be an int, the power up to which the components of vector x are going to be raised.
    Return:
        The matrix of polynomial features as a numpy.array, of dimension m * n,
            containing the polynomial feature values for all training examples.
        None if x is an empty numpy.array.
        None if x or power is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if x.__class__ != np.ndarray or power.__class__ != int:
        return None

    m = x.shape[0]
    features = np.zeros((m, power))

    for i in range(1, power + 1):
        features[:, i - 1] = x.ravel() ** i

    return features.astype(int)

if __name__ == "__main__":
    x = np.arange(1,6).reshape(-1, 1)

    # Example 0:
    print(f"<- {add_polynomial_features(x, 3)}")

    expected = np.array([
        [  1,   1,   1],
        [  2,   4,   8],
        [  3,   9,  27],
        [  4,  16,  64],
        [  5,  25, 125]
    ])
    print(f"-> {expected}")
    print()

    # Example 1:
    print(f"<- {add_polynomial_features(x, 6)}")

    expected = np.array([
        [    1,     1,     1,     1,     1,     1],
        [    2,     4,     8,    16,    32,    64],
        [    3,     9,    27,    81,   243,   729],
        [    4,    16,    64,   256,  1024,  4096],
        [    5,    25,   125,   625,  3125, 15625]
    ])
    print(f"-> {expected}")
