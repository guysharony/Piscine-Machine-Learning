import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def accuracy_score_(y, y_hat):
    """
    Compute the accuracy score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
    Returns:
        The accuracy score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if y.size == 0 or y_hat.size == 0:
        return None

    return np.mean(y == y_hat)


def precision_score_(y, y_hat, pos_label=1):
    """
    Compute the precision score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
        The precision score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if y.size == 0 or y_hat.size == 0:
        return None

    true_positives = np.sum((y == pos_label) & (y_hat == pos_label))
    false_positives = np.sum((y != pos_label) & (y_hat == pos_label))

    return true_positives / (true_positives + false_positives)


def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
        The recall score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if y.size == 0 or y_hat.size == 0:
        return None

    true_positives = np.sum((y == pos_label) & (y_hat == pos_label))
    false_negatives = np.sum((y == pos_label) & (y_hat != pos_label))

    return true_positives / (true_positives + false_negatives)


def f1_score_(y, y_hat, pos_label=1):
    """
    Compute the f1 score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
        The f1 score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if y.size == 0 or y_hat.size == 0:
        return None

    precision = precision_score_(y, y_hat, pos_label)
    recall = recall_score_(y, y_hat, pos_label)

    if precision is None or recall is None:
        return None

    return (2 * precision * recall) / (precision + recall)


if __name__ == "__main__":
    # Example 1:
    y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))
    
    # Accuracy
    print(f"<- {accuracy_score_(y, y_hat)}")
    print(f"-> {accuracy_score(y, y_hat)}")
    print()

    # Precision
    print(f"<- {precision_score_(y, y_hat)}")
    print(f"-> {precision_score(y, y_hat)}")
    print()

    # Recall
    print(f"<- {recall_score_(y, y_hat)}")
    print(f"-> {recall_score(y, y_hat)}")
    print()

    # F1-score
    print(f"<- {f1_score_(y, y_hat)}")
    print(f"-> {f1_score(y, y_hat)}")
    print()


    # Example 2:
    y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])

    # Accuracy
    print(f"<- {accuracy_score_(y, y_hat)}")
    print(f"-> {accuracy_score(y, y_hat)}")
    print()

    # Precision
    print(f"<- {precision_score_(y, y_hat, pos_label='dog')}")
    print(f"-> {precision_score(y, y_hat, pos_label='dog')}")
    print()

    # Recall
    print(f"<- {recall_score_(y, y_hat, pos_label='dog')}")
    print(f"-> {recall_score(y, y_hat, pos_label='dog')}")
    print()

    # F1-score
    print(f"<- {f1_score_(y, y_hat, pos_label='dog')}")
    print(f"-> {f1_score(y, y_hat, pos_label='dog')}")
    print()

    # Example 3:
    y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])

    # Precision
    print(f"<- {precision_score_(y, y_hat, pos_label='norminet')}")
    print(f"-> {precision_score(y, y_hat, pos_label='norminet')}")
    print()

    # Recall
    print(f"<- {recall_score_(y, y_hat, pos_label='norminet')}")
    print(f"-> {recall_score(y, y_hat, pos_label='norminet')}")
    print()

    # F1-score
    print(f"<- {f1_score_(y, y_hat, pos_label='norminet')}")
    print(f"-> {f1_score(y, y_hat, pos_label='norminet')}")