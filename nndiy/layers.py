from nndiy.linalg import Number, Vector, Matrix2D
from nndiy.activation import ActivationFunction, Identity


class Layer:
    weights: Vector
    biases: Vector
    values: Vector
    activations: Vector

    def __init__(self, size: Number, activation_function: ActivationFunction):
        self.size = size
        self.activation_function = activation_function


class HiddenLayer(Layer):
    def __init__(self, size: Number, activation_function: ActivationFunction):
        super().__init__(size, activation_function)


# supports only a single vector
class InputLayer(Layer):
    def __init__(self, input: Matrix2D, labels: Matrix2D):
        super().__init__(input.dimensions[0], Identity())
        self.train_samples = input
        self.train_labels = labels
        self.sample_id = 0
        self.values = input.columns[self.sample_id]  # set initial values to first row
        self.activations = self.values
        self.weights = None
        self.biases = None

    def get_sample(self, sample_id: int = None):
        if sample_id is None:
            sample_id = self.sample_id
        return self.train_samples.columns[sample_id], self.train_labels.columns[sample_id]


# supports only single output
class OutputLayer(Layer):
    def __init__(self, size: int, activation_function: ActivationFunction):
        super().__init__(size, activation_function)
