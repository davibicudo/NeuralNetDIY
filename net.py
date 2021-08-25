from typing import List
from layers import InputLayer, OutputLayer, Layer


class NeuralNet:
    def __init__(self, layers: List['Layer'], learning_rate=1):
        self.layers = layers
        self.learning_rate = learning_rate
        assert isinstance(layers[0], InputLayer)
        assert isinstance(layers[-1], OutputLayer)

    def initialize_params(self):
        pass


if __name__ == '__main__':
    pass
    # inputs = []
    # outputs = []
    # train_size = int(len(inputs)*.8)
    # train_images, test_images = Vector(inputs[:train_size]), Vector(inputs[train_size:])
    # train_labels, test_labels = Vector(outputs[:train_size]), Vector(outputs[train_size:])


