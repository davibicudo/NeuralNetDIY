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
    def exp(self, value):
        # avoid numerical overflow by limiting exponent to 100 (absolute)
        exp_100 = 2.6881171418161356e+43
        exp_n100 = 3.720075976020836e-44
        if abs(value) < 100:
            return math.exp(value)
        elif value > 0:
            return exp_100
        else:
            return exp_n100

    def calc(self, inputs: Vector) -> Vector:
        return Vector([1.0 / (1.0 + self.exp(-i)) for i in inputs.vector])

    def d_calc(self, inputs: Vector) -> Vector:
        return Vector([self.exp(-i) / (1.0 + self.exp(-i))**2 for i in inputs.vector])


class Softmax(ActivationFunction):
    def calc(self, inputs: Vector) -> Vector:
        return Vector([math.exp(i)/sum([math.exp(i_) for i_ in inputs.vector]) for i in inputs.vector])

    def d_calc(self, inputs: Vector) -> Vector:
        raise NotImplementedError('Non-trivial!! '
                                  'See: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/')

