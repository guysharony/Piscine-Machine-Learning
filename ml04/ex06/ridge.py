import numpy as np
from mylinearregression import MyLinearRegression

class MyRidge(MyLinearRegression):
    """
    Description:
        My personnal ridge regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        super().__init__(thetas, alpha, max_iter)
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
        self.lambda_ = lambda_

    def get_params_(self):
        return vars(self)

    def set_params_(self, **params):
        for key, value in params.items():
            if key in self.get_params_().keys():
                setattr(self, key, value)

    def l2(self, theta):
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

    def loss_(self, y, y_hat, theta, lambda_):
        """Computes the regularized loss of a linear regression model from two non-empty numpy.array, without any for loop.
        Args:
            y: has to be an numpy.ndarray, a vector of shape m * 1.
            y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
            theta: has to be a numpy.ndarray, a vector of shape n * 1.
            lambda_: has to be a float.
        Returns:
            The regularized loss as a float.
            None if y, y_hat, or theta are empty numpy.ndarray.
            None if y and y_hat do not share the same shapes.
        Raises:
            This function should not raise any Exception.
        """
        if y.__class__ != np.ndarray or y_hat.__class__ != np.ndarray or theta.__class__ != np.ndarray:
            return None

        if y.size == 0 or y_hat.size == 0 or theta.size == 0:
            return None

        if y.shape != y_hat.shape:
            return None

        m = y.shape[0]
        loss = (1 / (2 * m)) * (np.dot((y_hat - y).T, y_hat - y) + lambda_ * self.l2(theta))

        return loss[0][0].astype(float)

    def gradient_(self, y, x, theta):
        """Computes the regularized linear gradient of three non-empty numpy.ndarray,
            without any for-loop. The three arrays must have compatible shapes.
        Args:
            y: has to be a numpy.ndarray, a vector of shape m * 1.
            x: has to be a numpy.ndarray, a matrix of dimesion m * n.
            theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
            lambda_: has to be a float.
        Return:
            A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
            None if y, x, or theta are empty numpy.ndarray.
            None if y, x or theta does not share compatibles shapes.
            None if y, x or theta or lambda_ is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        if x.__class__ != np.ndarray or y.__class__ != np.ndarray or theta.__class__ != np.ndarray:
            return None

        if x.size == 0 or y.size == 0 or theta.size == 0:
            return None
        
        if y.shape[0] != x.shape[0]:
            return None

        if x.shape[1] + 1 != theta.shape[0]:
            return None

        m = x.shape[0]
        x_prime = np.hstack((np.ones((m, 1)), x))
        theta_prime = theta.copy()
        theta_prime[0] = 0

        gradient = (1 / m) * (
            np.dot(x_prime.T, (np.dot(x_prime, theta) - y)) + (self.lambda_ * theta_prime)
        )

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
            gradient = np.clip(self.gradient_(x, y, self.thetas), -1e8, 1e8)
            self.thetas -= self.alpha * gradient
        return self.thetas
