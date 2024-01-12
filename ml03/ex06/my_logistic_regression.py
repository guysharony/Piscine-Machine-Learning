import numpy as np

class MyLogisticRegression():
    """
    Description:
        My personnal logistic regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=100000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    @staticmethod
    def sigmoid_(x):
        if x.size == 0:
            return None

        return 1 / (1 + np.exp(-x))

    def gradient_(self, x, y):
        if x.__class__ != np.ndarray or y.__class__ != np.ndarray or self.thetas.__class__ != np.ndarray:
            return None

        if x.shape[1] + 1 != self.thetas.shape[0]:
            return None

        if x.size == 0 or y.size == 0 or self.thetas.size == 0:
            return None

        m = x.shape[0]

        x_prime = np.hstack((np.ones((m, 1)), x))
        y_hat = self.predict_(x)
        return x_prime.T.dot(y_hat - y) / m

    def predict_(self, x):
        if x.__class__ != np.ndarray or self.thetas.__class__ != np.ndarray:
            return None

        if x.size == 0 or self.thetas.size == 0:
            return None

        if x.shape[1] + 1 != self.thetas.shape[0]:
            return None

        x_prime = np.hstack((np.ones((x.shape[0], 1)), x))
        return MyLogisticRegression.sigmoid_(np.dot(x_prime, self.thetas))

    def loss_(self, y, y_hat):
        eps = 1e-15

        if y.__class__ != np.ndarray or y_hat.__class__ != np.ndarray:
            return None

        if y.size == 0 or y_hat.size == 0:
            return None

        m = y.shape[0]
        ones_vector = np.ones((m, 1))
        return - (1 / m) * np.sum(
            y * np.log(y_hat + eps) + (ones_vector - y) * np.log(ones_vector - y_hat + eps)
        )

    def fit_(self, x, y):
        if x.__class__ != np.ndarray or y.__class__ != np.ndarray or self.thetas.__class__ != np.ndarray:
            return None

        if x.size == 0 or self.thetas.size == 0:
            return None

        if self.thetas.shape[0] != x.shape[1] + 1 or x.shape[0] != y.shape[0]:
            return None

        for _ in range(self.max_iter):
            gradient = self.gradient_(x, y)
            self.thetas -= self.alpha * gradient
        return self.thetas

if __name__ == "__main__":
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
    Y = np.array([[1], [0], [1]])
    thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
    mylr = MyLogisticRegression(thetas)

    # Example 0:
    y_prediction = mylr.predict_(X)
    print(f"<- {y_prediction}")
    print(f"-> {np.array([[0.99930437], [1.0], [1.0]])}")
    print()

    # Example 1:
    print(f"<- {mylr.loss_(Y, y_prediction)}")
    print(f"-> {11.513157421577004}")
    print()

    # Example 2:
    mylr.fit_(X, Y)
    print(f"<- {mylr.thetas}")
    print(f"-> {np.array([[2.11826435], [0.10154334], [6.43942899], [-5.10817488], [0.6212541]])}")
    print()

    # Example 3:
    y_prediction = mylr.predict_(X)
    print(f"<- {y_prediction}")
    print(f"-> {np.array([[0.57606717], [0.68599807], [0.06562156]])}")
    print()

    # Example 4:
    print(f"<- {mylr.loss_(Y, y_prediction)}")
    print(f"-> {1.4779126923052268}")