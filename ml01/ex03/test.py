import numpy as np
from my_linear_regression import MyLinearRegression as MyLR

if __name__ == "__main__":
    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
    lr1 = MyLR(np.array([[2], [0.7]]))

    # Example 0.0:
    y_hat = lr1.predict_(x)
    print(f"<- {y_hat}")
    print(f"-> {np.array([[10.74695094], [17.05055804], [24.08691674], [36.24020866], [42.25621131]])}")
    print()

    # Example 0.1:
    print(f"<- {lr1.loss_elem_(y, y_hat)}")
    print(f"-> {np.array([[710.45867381], [364.68645485], [469.96221651], [108.97553412], [299.37111101]])}")
    print()

    # Example 0.2:
    print(f"<- {lr1.loss_(y, y_hat)}")
    print(f"-> {195.34539903032385}")
    print()

    # Example 1.0:
    lr2 = MyLR(np.array([[1], [1]]), 5e-8, 1500000)
    lr2.fit_(x, y)
    print(f"<- {lr2.thetas}")
    print(f"-> {np.array([[1.40709365], [1.1150909 ]])}")
    print()

    # Example 1.1:
    y_hat = lr2.predict_(x)
    print(f"<- {y_hat}")
    print(f"-> {np.array([[15.3408728 ], [25.38243697], [36.59126492], [55.95130097], [65.53471499]])}")
    print()

    # Example 1.2:
    print(f"<- {lr2.loss_elem_(y, y_hat)}")
    print(f"-> {np.array([[486.66604863], [115.88278416], [ 84.16711596], [ 85.96919719], [ 35.71448348]])}")
    print()

    # Example 1.3:
    print(f"<- {lr2.loss_(y, y_hat)}")
    print(f"-> {80.83996294128525}")