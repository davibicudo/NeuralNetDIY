from typing import List
import random
import time

from nndiy.cost import CostFunction
from nndiy.layers import InputLayer, OutputLayer, HiddenLayer, Layer
from nndiy.linalg import Vector, Matrix2D, DifferentDimensionError


class NeuralNet:
    def __init__(self, input_layer: InputLayer, output_layer: OutputLayer,
                 hidden_layers: List[HiddenLayer], cost_function: CostFunction, learning_rate=1e-4):
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
            self.layers[i].biases = Vector([0]*self.layers[i].size)

    def predict(self, data):
        self.input_layer.values = data
        for i in range(1, len(self.layers)):
            self.layers[i].values = (self.layers[i].weights
                                     .product(self.layers[i - 1].activations)
                                     .sum(self.layers[i].biases))
            self.layers[i].activations = self.layers[i].activation_function.calc(self.layers[i].values)

        return self.output_layer.values

    def compute_sample(self, sample_data, sample_label):
        predicted = self.predict(sample_data)
        return self.cost_function.calc(predicted, sample_label)

    def backpropagate_sample(self, sample_id: int = 0):
        # backpropagates through whole network for one sample and returns the sample gradients for each layer
        # sample_id is the column from train_samples to be calculated
        assert sample_id in range(self.input_layer.train_samples.dimensions[1])
        sample_data, sample_label = self.input_layer.get_sample(sample_id)
        loss = self.compute_sample(sample_data, sample_label)

        gradients_w = {}
        gradients_b = {}

        # calculate errors in output layer (BP1)
        current_err = self.cost_function.jacobian_calc(self.output_layer.activations, sample_label)
        current_err = current_err.hadamard(self.output_layer.activation_function.d_calc(self.output_layer.values))
        for i in range(len(self.layers) - 1, 0, -1):  # loop from output to first hidden layer
            current = self.layers[i]
            previous = self.layers[i - 1]

            # ensure dimensions
            assert current.size == current_err.size, DifferentDimensionError.raize()
            assert previous.activations.size == current.weights.dimensions[1], DifferentDimensionError.raize()
            assert current_err.size == current.weights.dimensions[0], DifferentDimensionError.raize()

            # update biases (BP3)
            gradient_b = current_err.scalar_prod(self.learning_rate)
            gradients_b[i] = gradient_b

            # update weights (BP4)
            gradient_w = previous.activations.product(current_err.T()).T().scalar_prod(self.learning_rate)
            gradients_w[i] = gradient_w

            # calculate error of previous layer based on current's (BP2)
            bp2_1 = current.weights.T().product(current_err)
            bp2_2 = previous.activation_function.d_calc(previous.values)
            current_err = bp2_1.hadamard(bp2_2)

        return gradients_w, gradients_b, loss

    def train_epoch(self):
        # trains with all data
        all_gradients_w = []
        all_gradients_b = []
        losses = []
        for sample_id in range(self.input_layer.train_samples.dimensions[1]):
            gradients_w, gradients_b, loss = self.backpropagate_sample(sample_id)
            all_gradients_w.append(gradients_w)
            all_gradients_b.append(gradients_b)
            losses.append(loss)

        # average all gradients and apply to each layer
        for i in range(1, len(self.layers)):
            layer_gradient_w = [g_w[i] for g_w in all_gradients_w]
            gradients_w = sum(layer_gradient_w)
            gradients_w = gradients_w.scalar_prod(1/len(all_gradients_w))
            gradients_b = sum([g_b[i] for g_b in all_gradients_b])
            gradients_b = gradients_b.scalar_prod(1 / len(all_gradients_b))

            self.layers[i].biases = self.layers[i].biases.sum(gradients_b)
            self.layers[i].weights = self.layers[i].weights.sum(gradients_w)

        # return epoch's loss
        return sum(losses)/len(losses)

    def train(self, loss=0.1, max_epochs=100):
        current_loss = 1e9
        epoch = 1
        while current_loss > loss and epoch <= max_epochs:
            start = time.time()
            current_loss = self.train_epoch()
            print(f"Finished Epoch {epoch} in {time.time() - start:.2f} seconds. Current loss: {current_loss:.3f}")
            epoch += 1





