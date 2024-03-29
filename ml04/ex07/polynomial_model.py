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

    m, n = x.shape
    features = np.zeros((m, n * power))

    for i in range(n):
        for j in range(1, power + 1):
            features[:, i * power + j - 1] = x[:, i] ** j

    return features.astype(int)
