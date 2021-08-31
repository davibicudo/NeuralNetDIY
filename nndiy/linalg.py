from typing import List, Union, TypeVar

Number = TypeVar('Number', int, float)


class DifferentDimensionError(Exception):
    @staticmethod
    def raize():
        raise DifferentDimensionError


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
        assert self.dimensions[0] == transposed.dimensions[1], DifferentDimensionError.raize()
        assert self.dimensions[1] == transposed.dimensions[0], DifferentDimensionError.raize()
        return transposed

    def product(self, other: 'Matrix2D'):
        return product(self, other)

    def sum(self, other: 'Matrix2D'):
        return sum_matrix(self, other)

    def __add__(self, other):
        return sum_matrix(self, other)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def hadamard(self, other: 'Matrix2D'):
        return hadamard_matrix(self, other)

    def scalar_prod(self, scalar: Number):
        new_columns = []
        for col in self.columns:
            new_columns.append(col.scalar_prod(scalar))
        return Matrix2D(new_columns)

    def scalar_sum(self, scalar: Number):
        new_columns = []
        for col in self.columns:
            new_columns.append(Vector([i+scalar for i in col.vector]))
        return Matrix2D(new_columns)

    def rr(self):
        r = ''
        for i in range(self.dimensions[0]):
            r += '|'
            for j in range(self.dimensions[1]):
                r += str(self.columns[j].vector[i])
                r += '  ' if j + 1 < self.dimensions[1] else '|'
            r += '\n'
        return r


class Vector(Matrix2D):
    def __init__(self, list_: List[Number]):
        super().__init__([self])
        self.vector = list_
        self.size = len(list_)
        self.dimensions = (self.size, 1)

    def dot(self, other):
        return dot(self, other)

    def scalar_prod(self, scalar: Number):
        return Vector([v*scalar for v in self.vector])

    def __repr__(self):
        return '(' + '  '.join([f'{i:.2f}' if isinstance(i, float) else str(i) for i in self.vector]) + ')'


def dot(v1: Vector, v2: Vector) -> Number:
    if v1.size != v2.size:
        raise DifferentDimensionError
    return sum([v1.vector[i]*v2.vector[i] for i in range(v1.size)])


def hadamard_vector(v1: Vector, v2: Vector) -> Vector:
    assert v1.dimensions == v2.dimensions, DifferentDimensionError.raize()
    return Vector([v1.vector[i] * v2.vector[i] for i in range(v1.size)])


def sum_matrix(m1: Matrix2D, m2: Matrix2D) -> Union[Matrix2D, Vector]:
    assert m1.dimensions == m2.dimensions, DifferentDimensionError.raize()
    new_columns = []
    for c1, c2 in zip(m1.columns, m2.columns):
        new_columns.append(
            Vector([c1.vector[i] + c2.vector[i] for i in range(c1.size)])
        )
    if len(new_columns) == 1:
        return new_columns[0]
    else:
        return Matrix2D(new_columns)


def product(m1: Matrix2D, m2: Matrix2D) -> Union[Number, Vector, Matrix2D]:
    assert m1.dimensions[1] == m2.dimensions[0], DifferentDimensionError.raize()
    # dot product of each column of m2 with each row of m1
    # for convenience we take the transposed of m1
    m1t = m1.T()
    if (m2.dimensions[0] == 1 and m1.dimensions[1] == 1) and (m2.dimensions[1] == m1.dimensions[0]):
        return dot(m1.columns[0], m2.columns[0])
    elif m2.dimensions[1] == 1:
        return Vector([dot(m2.columns[0], r) for r in m1t.columns])
    else:
        new_columns = []
        for c in m2.columns:
            new_columns.append(Vector([dot(c, r) for r in m1t.columns]))
        new_matrix = Matrix2D(new_columns)
        assert new_matrix.dimensions == (m1.dimensions[0], m2.dimensions[1]), DifferentDimensionError.raize()
        return new_matrix


def hadamard_matrix(m1: Matrix2D, m2: Matrix2D) -> Union[Matrix2D, Vector]:
    assert m1.dimensions == m2.dimensions, DifferentDimensionError.raize()
    new_columns = []
    for c1, c2 in zip(m1.columns, m2.columns):
        new_columns.append(hadamard_vector(c1, c2))
    if len(new_columns) == 1:
        return new_columns[0]
    else:
        return Matrix2D(new_columns)

