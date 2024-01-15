import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def confusion_matrix_(y_true, y_hat, labels=None, df_option=False):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
        labels: optional, a list of labels to index the matrix.
          This may be used to reorder or select a subset of labels. (default=None)
        df_option: optional, if set to True the function will return a pandas DataFrame
          instead of a numpy array. (default=False)
    Return:
        The confusion matrix as a numpy array or a pandas DataFrame according to df_option value.
        None if any error.
    Raises:
        This function should not raise any Exception.
    """
    y_true = y_true.flatten()
    y_hat = y_hat.flatten()

    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_hat)))

    num_labels = len(labels)
    confusion_matrix = np.zeros((num_labels, num_labels), dtype=int)

    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            confusion_matrix[i, j] = np.sum((y_true == true_label) & (y_hat == pred_label))

    if df_option:
        return pd.DataFrame(confusion_matrix, index=labels, columns=labels)

    return confusion_matrix

if __name__ == "__main__":
    y_hat = np.array([['norminet'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['bird']])
    y = np.array([['dog'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['norminet']])

    # Example 1:
    print(f"<- {confusion_matrix_(y, y_hat)}")
    print(f"-> {confusion_matrix(y, y_hat)}")
    print()

    # Example 2:
    print(f"<- {confusion_matrix_(y, y_hat, labels=['dog', 'norminet'])}")
    print(f"-> {confusion_matrix(y, y_hat, labels=['dog', 'norminet'])}")

    # Example 3:
    print(confusion_matrix_(y, y_hat, df_option=True))
    print()

    # Example 4:
    print(confusion_matrix_(y, y_hat, labels=['bird', 'dog'], df_option=True))
    print()