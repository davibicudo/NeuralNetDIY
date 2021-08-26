from typing import List
from random import random

from cost import CostFunction
from layers import InputLayer, OutputLayer, HiddenLayer, Layer
from linalg import Vector, Matrix2D


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
                [Vector([random()] * self.layers[i].size) for _ in range(self.layers[i-1].size)])
            self.layers[i].biases = Vector([random()] * self.layers[i].size)

    def compute_single(self):
        for i in range(1, len(self.layers)):
            self.layers[i].values = (self.layers[i].weights
                                     .product(self.layers[i-1].values)
                                     .sum(self.layers[i].biases))
            self.layers[i].values = self.layers[i].activation.calc(self.layers[i].values)

    def compute_cost(self, labels: Vector):
        return self.cost_function.calc(self.output_layer.values, labels)

