import numpy as np

class MyLinearRegression():
    """
    Description:
        My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=10000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    @staticmethod
    def gradient(x, y, theta):
        """Computes a gradient vector from three non-empty numpy.array, without any for loop.
            The three arrays must have compatible shapes.
        Args:
            x: has to be a numpy.array, a matrix of shape m * 1.
            y: has to be a numpy.array, a vector of shape m * 1.
            theta: has to be a numpy.array, a 2 * 1 vector.
        Return:
            The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
            None if x, y, or theta is an empty numpy.ndarray.
            None if x, y and theta do not have compatible dimensions.
        Raises:
            This function should not raise any Exception.
        """
        if y.__class__ != np.ndarray or x.__class__ != np.ndarray or theta.__class__ != np.ndarray:
            return None

        if y.size == 0 or x.size == 0 or theta.size == 0:
            return None

        m = x.shape[0]
        x_prime = np.c_[np.ones(m), x]
        diff = np.dot(x_prime, theta) - y
        gradient = np.dot(x_prime.T, diff) / m
        return gradient

    def fit_(self, x, y):
        """
        Description:
            Fits the model to the training dataset contained in x and y.
        Args:
            x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
            y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
            theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
            alpha: has to be a float, the learning rate
            max_iter: has to be an int, the number of iterations done during the gradient descent
        Returns:
            new_theta: numpy.ndarray, a vector of dimension 2 * 1.
            None if there is a matching dimension problem.
        Raises:
            This function should not raise any Exception.
        """
        if x.__class__ != np.ndarray or y.__class__ != np.ndarray or self.thetas.__class__ != np.ndarray:
            return None

        if x.size == 0 or self.thetas.size == 0:
            return None

        if self.thetas.shape[0] != x.shape[1] + 1 or x.shape[0] != y.shape[0]:
            return None

        self.thetas = self.thetas.astype(float).copy()
        for _ in range(self.max_iter):
            gradient = MyLinearRegression.gradient(x, y, self.thetas)
            self.thetas -= self.alpha * gradient
        return self.thetas

    def loss_(self, y, y_hat):
        """
        Description:
            Calculates the value of loss function.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            J_value : has to be a float.
            None if there is a dimension matching problem between X, Y or theta.
            None if any argument is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        if y.__class__ != np.ndarray or y_hat.__class__ != np.ndarray:
            return None

        if y.size != y_hat.size:
            return None

        if y.size == 0 or y_hat.size == 0:
            return None

        return np.sum(self.loss_elem_(y, y_hat)) / (2 * y.shape[0])

    def loss_elem_(self, y, y_hat):
        """
        Description:
            Calculates all the elements (y_pred - y)^2 of the loss function.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            J_elem: numpy.array, a vector of dimension (number of the training examples,1).
            None if there is a dimension matching problem between X, Y or theta.
            None if any argument is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        if y.__class__ != np.ndarray or y_hat.__class__ != np.ndarray:
            return None

        if y.size != y_hat.size:
            return None

        if y.size == 0 or y_hat.size == 0:
            return None

        return (y_hat - y) ** 2

    def predict_(self, x):
        return np.dot(np.c_[np.ones(x.shape[0]), x], self.thetas)

    @staticmethod
    def mse_(y, y_hat):
        if y.shape != y_hat.shape:
            return None
        return np.mean((y_hat - y) ** 2)
