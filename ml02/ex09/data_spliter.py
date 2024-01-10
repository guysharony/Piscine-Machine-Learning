import numpy as np

def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y) into a training and a test set,
        while respecting the given proportion of examples to be kept in the training set.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        proportion: has to be a float, the proportion of the dataset that will be assigned to the
            training set.
    Return:
        (x_train, x_test, y_train, y_test) as a tuple of numpy.array
        None if x or y is an empty numpy.array.
        None if x and y do not share compatible dimensions.
        None if x, y or proportion is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if x.__class__ != np.ndarray or y.__class__ != np.ndarray or proportion.__class__ != float:
        return None

    if x.size == 0 or y.size == 0:
        return None

    if x.shape[0] != y.shape[0]:
        return None

    dataset = np.hstack((x, y))
    np.random.shuffle(dataset)

    x_dataset = dataset[:, :-1]
    y_dataset = dataset[:, -1].reshape(-1, 1)

    split_index = int(x.shape[0] * proportion)

    x_train, x_test = x_dataset[:split_index], x_dataset[split_index:]
    y_train, y_test = y_dataset[:split_index], y_dataset[split_index:]

    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    x1 = np.array([1, 42, 300, 10, 59]).reshape((-1, 1))
    y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))
    
    # Example 1:
    print(f"<- {data_spliter(x1, y, 0.8)}")
    print("-> (array([ 1, 59, 42, 300]), array([10]), array([0, 0, 1, 0]), array([1]))")
    print()

    # Example 2:
    print(f"<- {data_spliter(x1, y, 0.5)}")
    print("-> (array([59, 10]), array([ 1, 300, 42]), array([0, 1]), array([0, 0, 1])))")
    print()
    
    x2 = np.array([[  1, 42],
                [300, 10],
                [ 59,  1],
                [300, 59],
                [ 10, 42]])
    y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))

    # Example 3:
    print(f"<- {data_spliter(x2, y, 0.8)}")
    print("-> (array([[ 10, 42], [300,  59], [ 59,   1], [300,  10]])), array([[ 1, 42]]), array([0, 1, 0, 1]), array([0]))")
    print()

    # Example 4:
    print(f"<- {data_spliter(x2, y, 0.5)}")
    print("-> array([[59, 1], [10, 42]]), array([[300,  10], [300,  59], [  1,  42]]), array([0, 0]), array([1, 1, 0]))")