from matrix import Matrix
from matrix import Vector

if __name__ == '__main__':
    m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
    print("matrix 1:")
    print(m1.shape)
    print(m1.data)
    print(m1.T())

    m2 = Matrix([[5.0, 6.0], [7.0, 8.0]])
    print()
    print("matrix 2:")
    print(m2.shape)
    print(m2.data)
    print(m2.T())

    print()
    print('m1 + m2: ', m1 + m2)
    print('m1 - m2: ', m1 - m2)
    print('m1 / 5: ', m1 / 5)