from dataclasses import dataclass, field

@dataclass
class Matrix:
    data = field(default_factory=list(list(int, float)))
    shape = field(default_factory=tuple(int, int))

    def __init__(self, values):
        if values.__class__ not in (list, tuple):
            raise ValueError("Matrix must initialize with tuple or list.")

        if values.__class__ == list:
            if not all(len(row) == len(values[0]) for row in values):
                raise ValueError("Matrix rows must be the same size.")

            if not all(value.__class__ in (int, float) for row in values for value in row):
                raise ValueError("Matrix values must be int or floats.")

            self.data = values
            self.shape = (len(values), len(values[0]))

        if values.__class__ == tuple:
            if len(values) != 2:
                raise ValueError("Shape of matrix contain 2 integers.")

            if not all(value.__class__ in (int) for value in values):
                raise ValueError("Shape of matrix contain 2 integers.")

            self.data = [[0] * values[1] for _ in range(values[0])]
            self.shape = values

    def __add__(self, m: 'Matrix') -> 'Matrix':
        if self.shape != m.shape:
            raise ValueError("Matrices must have the same dimensions for addition.")

        self.data = [[x + y for x, y in zip(a, b)] for a, b in zip(self.data, m)]
        return self

    def __radd__(self, m: 'Matrix') -> 'Matrix':
        return self.__add__(m)

    def __sub__(self, m: 'Matrix') -> 'Matrix':
        if self.shape != m.shape:
            raise ValueError("Matrices must have the same dimensions for subtraction.")

        self.data = [[x - y for x, y in zip(a, b)] for a, b in zip(self.data, m)]
        return self

    def __rsub__(self, m: 'Matrix') -> 'Matrix':
        return self.__sub__(m)