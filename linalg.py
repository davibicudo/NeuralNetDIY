from typing import List, Union, TypeVar

Number = TypeVar('Number', int, float)


class DifferentDimensionError(Exception):
    pass


class Matrix2D:
    def __init__(self, vectors: List['Vector']):
        self.columns = vectors
        self.dimensions = (vectors[0].size, len(vectors))
        if not all([v.size == self.dimensions[0] for v in vectors]):
            raise DifferentDimensionError

    def T(self):
        new_columns = []
        for v in self.columns:
            new_vector = []
            for i in range(len(self.columns)):
                new_vector.append(v.vector[i])
            new_columns.append(Vector(new_vector))
        transposed = Matrix2D(new_columns)
        assert self.dimensions[0] == transposed.dimensions[1]
        assert self.dimensions[1] == transposed.dimensions[0]
        return transposed

    def product(self, other: 'Matrix2D'):
        return product(self, other)


class Vector(Matrix2D):
    def __init__(self, list_: List[Number]):
        super().__init__([self])
        self.vector = list_
        self.size = len(list_)

    def dot(self, other):
        return dot(self, other)


def dot(v1: Vector, v2: Vector) -> Number:
    if v1.size != v2.size:
        raise DifferentDimensionError
    return sum([v1.vector[i]*v2.vector[i] for i in range(v1.size)])


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



