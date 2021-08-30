import math

from nndiy.linalg import Vector


class ActivationFunction:
    def calc(self, inputs: Vector) -> Vector:
        pass

    def d_calc(self, inputs: Vector) -> Vector:
        pass


class Identity(ActivationFunction):
    def calc(self, inputs: Vector) -> Vector:
        return inputs

    def d_calc(self, inputs: Vector) -> Vector:
        return Vector([1]*inputs.size)


class ReLU(ActivationFunction):
    def calc(self, inputs: Vector) -> Vector:
        return Vector([max(0, i) for i in inputs.vector])

    def d_calc(self, inputs: Vector) -> Vector:
        return Vector([1 if i > 0 else 0 for i in inputs.vector])


class Sigmoid(ActivationFunction):
    def calc(self, inputs: Vector) -> Vector:
        return Vector([1.0 / (1.0 + math.exp(-i)) for i in inputs.vector])

    def d_calc(self, inputs: Vector) -> Vector:
        return Vector([math.exp(-i) / (1.0 + math.exp(-i))**2 for i in inputs.vector])


class Softmax(ActivationFunction):
    def calc(self, inputs: Vector) -> Vector:
        return Vector([math.exp(i)/sum([math.exp(i_) for i_ in inputs.vector]) for i in inputs.vector])

    def d_calc(self, inputs: Vector) -> Vector:
        raise NotImplementedError('Non-trivial!! '
                                  'See: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/')

