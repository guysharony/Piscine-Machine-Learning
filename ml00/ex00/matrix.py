from dataclasses import dataclass, field
from typing import Union

@dataclass
class Matrix:
    data: list = field(default_factory=list)
    shape: tuple = field(default_factory=tuple)

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

        return Matrix([
            [x + y for x, y in zip(a, b)]
            for a, b in zip(self.data, m.data)
        ])

    def __radd__(self, m: 'Matrix') -> 'Matrix':
        return m.__add__(self)

    def __sub__(self, m: 'Matrix') -> 'Matrix':
        if self.shape != m.shape:
            raise ValueError("Matrices must have the same dimensions for subtraction.")

        return Matrix([
            [x - y for x, y in zip(a, b)]
            for a, b in zip(self.data, m.data)
        ])

    def __rsub__(self, m: 'Matrix') -> 'Matrix':
        return m.__sub__(self)

    def __truediv__(self, s: Union[int, float]) -> 'Matrix':
        if s.__class__ not in (int, float):
            raise ValueError("Scalar value must be int or float.")

        return Matrix([
            [x / s for x in a]
            for a in self.data
        ])

    def __rtruediv__(self, s: float) -> 'Matrix':
        return self.__truediv__(1 / s)

    def __mul__(self, u: Union[float, 'Matrix']):
        if u.__class__ in (float, int):
            return Matrix([
                [x / u for x in a]
                for a in self.data
            ])

        if u.__class__ in (Matrix):
            return Matrix([
                [x * y for x, y in zip(a, b)]
                for a, b in zip(self.data, u.data)
            ])

        raise ValueError(f"Cannot compute multiplication with {type(u)}")

    def __rmul__(self, u: Union[float, 'Matrix']):
        return self.__mul__(u)

    def __str__(self):
        return f"{type(self)}: {str(self.data)}"

    def __repr__(self):
        return f"{type(self)}: {str(self.data)}"

    def T(self) -> 'Matrix':
        rows, columns = self.shape

        result = [[0] * columns for _ in range(rows)]
        for n in range(columns):
            for m in range(rows):
                result[n][m] = self.data[m][n]

        return Matrix(result)
    

class Vector(Matrix):
    def __init__(self, values):
        if not values.__class__ == list:
            raise ValueError("Vector must be a list.")

        if not (len(values) == 1 or all(len(row) == 1 for row in values)):
            raise ValueError("Vector sublists must contain no more than 1 value.")

        super().__init__(values)
