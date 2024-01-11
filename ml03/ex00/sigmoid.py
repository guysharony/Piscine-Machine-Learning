import numpy as np

def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    Args:
        x: has to be a numpy.ndarray of shape (m, 1).
    Returns:
        The sigmoid value as a numpy.ndarray of shape (m, 1).
        None if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if x.size == 0:
        return None

    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    # Example 1:
    x = np.array([[-4]])
    print(f"<- {sigmoid_(x)}")
    print(f"-> {np.array([[0.01798620996209156]])}")
    print()

    # Example 2:
    x = np.array([[2]])
    print(f"<- {sigmoid_(x)}")
    print(f"-> {np.array([[0.8807970779778823]])}")
    print()

    # Example 3:
    x = np.array([[-4], [2], [0]])
    print(f"<- {sigmoid_(x)}")
    print(f"-> {np.array([[0.01798620996209156], [0.8807970779778823], [0.5]])}")