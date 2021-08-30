from typing import List
import random

from nndiy.cost import CostFunction
from nndiy.layers import InputLayer, OutputLayer, HiddenLayer, Layer
from nndiy.linalg import Vector, Matrix2D


class NeuralNet:
    def __init__(self, input_layer: InputLayer, output_layer: OutputLayer,
                 hidden_layers: List[HiddenLayer], cost_function: CostFunction, learning_rate=1):
        self.learning_rate = learning_rate
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.hidden_layers = hidden_layers
        self.layers: List[Layer] = [input_layer] + hidden_layers + [output_layer]
        self.cost_function = cost_function

        # initialize weights and biases randomly
        for i in range(1, len(self.layers)):
            self.layers[i].weights = Matrix2D(
                [Vector([random.uniform(0, 1) for _ in range(self.layers[i].size)])
                 for __ in range(self.layers[i-1].size)])
            self.layers[i].biases = Vector([random.uniform(0, 1) for _ in range(self.layers[i].size)])

    def compute_single(self, sample_id: int = 0):
        # sample_id is the column from train_samples to be calculated
        assert sample_id in range(self.input_layer.train_samples.dimensions[1])
        self.input_layer.values = self.input_layer.train_samples.columns[sample_id]
        for i in range(1, len(self.layers)):
            self.layers[i].values = (self.layers[i].weights
                                     .product(self.layers[i-1].values)
                                     .sum(self.layers[i].biases))
            self.layers[i].activations = self.layers[i].activation_function.calc(self.layers[i].values)

    def compute_cost_simple(self, labels: Vector):
        return self.cost_function.calc(self.output_layer.values, labels)

    def backpropagate_single(self):
        self.compute_single()
        # calculate errors in output layer (BP1)
        current_err = self.cost_function.jacobian_calc(self.output_layer.activations, self.input_layer.get_sample()[0])
        current_err = current_err.hadamard(self.output_layer.activation_function.d_calc(self.output_layer.values))
        for i in range(len(self.layers), 0, -1):  # loop from output to first hidden layer
            current = self.layers[i]
            previous = self.layers[i - 1]

            # update biases (BP3)
            current.biases = current.biases.sum(current_err)

            # update weights (BP4)
            current.weights = current.weights.sum(previous.activations.product(current_err.T()))

            # calculate error of previous layer based on current's (BP2)
            bp2_1 = current.weights.T().product(current_err)
            bp2_2 = previous.activation_function.d_calc(previous.values)
            current_err = bp2_1.hadamard(bp2_2)

