from typing import List, Union, TypeVar

Number = TypeVar('Number', int, float)


class DifferentDimensionError(Exception):
    pass


class Matrix2D:
    def __init__(self, vectors: List['Vector']):
        self.dimension = None
        self.columns = vectors
        if not isinstance(self, Vector):
            self.dimensions = (vectors[0].size, len(vectors))
            if not all([v.size == self.dimensions[0] for v in vectors]):
                raise DifferentDimensionError

    def T(self):
        new_columns = []
        for i in range(self.dimensions[0]):
            new_vector = []
            for j in range(self.dimensions[1]):
                new_vector.append(self.columns[j].vector[i])
            new_columns.append(Vector(new_vector))
        transposed = Matrix2D(new_columns)
        assert self.dimensions[0] == transposed.dimensions[1]
        assert self.dimensions[1] == transposed.dimensions[0]
        return transposed

    def product(self, other: 'Matrix2D'):
        return product(self, other)

    def sum(self, other: 'Matrix2D'):
        return sum_matrix(self, other)

    def hadamard(self, other: 'Matrix2D'):
        return hadamard_matrix(self, other)

    def __add__(self, other: 'Matrix2D'):
        return sum_matrix(self, other)

    def __mul__(self, other: 'Matrix2D'):
        return product(self, other)


class Vector(Matrix2D):
    def __init__(self, list_: List[Number]):
        super().__init__([self])
        self.vector = list_
        self.size = len(list_)
        self.dimensions = (self.size, 1)

    def dot(self, other):
        return dot(self, other)

    def sum(self, other: 'Vector'):
        return sum_vector(self, other)

    def __add__(self, other: 'Vector'):
        return sum_vector(self, other)


def dot(v1: Vector, v2: Vector) -> Number:
    if v1.size != v2.size:
        raise DifferentDimensionError
    return sum([v1.vector[i]*v2.vector[i] for i in range(v1.size)])


def sum_vector(v1: Vector, v2: Vector) -> Vector:
    assert v1.dimensions == v2.dimensions
    return Vector([v1.vector[i] + v2.vector[i] for i in range(v1.size)])


def hadamard_vector(v1: Vector, v2: Vector) -> Vector:
    assert v1.dimensions == v2.dimensions
    return Vector([v1.vector[i] * v2.vector[i] for i in range(v1.size)])


def sum_matrix(m1: Matrix2D, m2: Matrix2D) -> Union[Matrix2D, Vector]:
    assert m1.dimensions == m2.dimensions
    new_columns = []
    for c1, c2 in zip(m1.columns, m2.columns):
        new_columns.append(sum_vector(c1, c2))
    if len(new_columns) == 1:
        return new_columns[0]
    else:
        return Matrix2D(new_columns)


def product(m1: Matrix2D, m2: Matrix2D) -> Union[Number, Vector, Matrix2D]:
    assert m1.dimensions[1] == m2.dimensions[0]
    # dot product of each column of m2 with each row of m1
    # for convenience we take the transposed of m1
    m1t = m1.T()
    if m2.dimensions[0] == 1 and m1.dimensions[1] == 1:
        return dot(m1.columns[0], m2.columns[0])
    elif m2.dimensions[1] == 1:
        return Vector([dot(m2.columns[0], r) for r in m1t.columns])
    else:
        new_columns = []
        for c in m2.columns:
            new_columns.append(Vector([dot(c, r) for r in m1t.columns]))
        new_matrix = Matrix2D(new_columns)
        assert new_matrix.dimensions == (m1.dimensions[0], m2.dimensions[1])
        return new_matrix


def hadamard_matrix(m1: Matrix2D, m2: Matrix2D) -> Union[Matrix2D, Vector]:
    assert m1.dimensions == m2.dimensions
    new_columns = []
    for c1, c2 in zip(m1.columns, m2.columns):
        new_columns.append(hadamard_vector(c1, c2))
    if len(new_columns) == 1:
        return new_columns[0]
    else:
        return Matrix2D(new_columns)

