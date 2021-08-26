from linalg import Number, Vector
from activation import ActivationFunction, Identity


class Layer:
    weights: Vector
    biases: Vector
    values: Vector

    def __init__(self, size: Number, activation: ActivationFunction):
        self.size = size
        self.activation = activation


class HiddenLayer(Layer):
    def __init__(self, size: Number, activation: ActivationFunction):
        super().__init__(size, activation)


# supports only a single vector
class InputLayer(Layer):
    def __init__(self, input: Vector):
        super().__init__(input.size, Identity())
        self.values = input
        self.weights = None
        self.biases = None


# supports only a single vector
class OutputLayer(Layer):
    def __init__(self, size: Number, activation: ActivationFunction):
        super().__init__(size, activation)
