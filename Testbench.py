#!/usr/bin/env python
# coding: utf-8

# [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/davibicudo/NeuralNetDIY/master)

# In[1]:


def get_ipython():
    dummy = type('', (), {})
    dummy.run_line_magic = lambda x, y: ''
    return dummy


# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from nndiy import utils
from nndiy import activation
from nndiy import cost
from nndiy import net
from nndiy import layers
from nndiy import linalg


# In[3]:

# test if linalg operations work correctly
# 2x2 identity
I2 = linalg.Matrix2D([
    linalg.Vector([1, 0]), linalg.Vector([0, 1])
])
v = linalg.Vector([2, 2])
assert I2.product(v).vector == [2, 2]
assert (I2 + I2).columns[0].vector == [2, 0]

# load data
data, labels = utils.read_data("MNIST/mnist_train_pt1.csv", nrows=10)
labels = linalg.Vector([int(i) for i in labels.vector])

# normalize data
data = data.scalar_prod(1/255)  # scalar_sum(-1)

# convert labels to matrix format as probabilities
new_columns = []
for label in labels.vector:
    new_columns.append(linalg.Vector([1 if label == i else 0 for i in range(10)]))
labels = linalg.Matrix2D(new_columns)

# In[5]:

# network for computing single sample
input_layer = layers.InputLayer(data, labels)
output_layer = layers.OutputLayer(size=10, activation_function=activation.Sigmoid())
hidden_layers = [
    layers.HiddenLayer(size=200, activation_function=activation.Sigmoid()),
    # layers.HiddenLayer(size=16, activation_function=activation.Sigmoid())
]
n = net.NeuralNet(input_layer=input_layer, output_layer=output_layer, hidden_layers=hidden_layers,
                  cost_function=cost.QuadraticDistance(), learning_rate=0.0001)


# In[ ]:


n.train()

data, label = n.input_layer.get_sample()
print(n.predict(data))
print(label)
