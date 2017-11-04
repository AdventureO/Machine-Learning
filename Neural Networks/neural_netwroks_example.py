from pybrain.tools.shortcuts import buildNetwork

# 2 - number of input layers, 3 - number of hidden layers, 1 - number of output layers
neural_network = buildNetwork(2, 3, 1)
# Activate neural network with input data 1 and 0
print(neural_network.activate([1, 0]))

# Print input, hidden and output layers
print(neural_network['in'])
print(neural_network['hidden0'])
print(neural_network['out'])