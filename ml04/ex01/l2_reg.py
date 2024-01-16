import numpy as np

def iterative_l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.
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

    l2_reg = 0.0
    for value in theta[1:]:
        l2_reg += value ** 2

    return l2_reg[0].astype(float)


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


if __name__ == "__main__":
    x = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))

    # Example 1:
    print(f"<- {iterative_l2(x)}")
    print(f"-> {911.0}")
    print()

    # Example 2:
    print(f"<- {l2(x)}")
    print(f"-> {911.0}")
    print()


    y = np.array([3, 0.5, -6]).reshape((-1, 1))

    # Example 3:
    print(f"<- {iterative_l2(y)}")
    print(f"-> {36.25}")
    print()

    # Example 4:
    print(f"<- {l2(y)}")
    print(f"-> {36.25}")