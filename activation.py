import math

from linalg import Vector


class ActivationFunction:
    def calc(self, inputs: Vector) -> Vector:
        pass


class Identity(ActivationFunction):
    def calc(self, inputs: Vector) -> Vector:
        return inputs


class ReLU(ActivationFunction):
    def calc(self, inputs: Vector) -> Vector:
        return Vector([max(0, i) for i in inputs.vector])


class Sigmoid(ActivationFunction):
    def calc(self, inputs: Vector) -> Vector:
        return Vector([1.0 / (1.0 + math.exp(-i)) for i in inputs.vector])


class Softmax(ActivationFunction):
    def calc(self, inputs: Vector) -> Vector:
        return Vector([math.exp(i)/sum([math.exp(i_) for i_ in inputs.vector]) for i in inputs.vector])
