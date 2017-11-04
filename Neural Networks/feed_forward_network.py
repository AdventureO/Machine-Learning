from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit

# Create Feed forward neural network
network = FeedForwardNetwork()

inputLayer = LinearLayer(2) # Set 2 linear input layers
hiddenLayer = SigmoidLayer(3) # Set 3 hidden layer with sigmoid function
outputLayer = SigmoidLayer(1) # Set 1 output layer with sigmoid function

# Create bias units
bias1 = BiasUnit()
bias2 = BiasUnit()

# Add to our network bias units
network.addModule(bias1)
network.addModule(bias2)

# Add each layer of neurons we created to our model
network.addInputModule(inputLayer)
network.addModule(hiddenLayer)
network.addOutputModule(outputLayer)

# Create full connections between our layers
inputHidden = FullConnection(inputLayer, hiddenLayer)
hiddenOutput = FullConnection(hiddenLayer, outputLayer)

# Add bias dependency to our layers
biasToHidden = FullConnection(bias1, hiddenLayer)
biasToOutput = FullConnection(bias2, outputLayer)

# Add layers connections to our network
network.addConnection(inputHidden)
network.addConnection(hiddenOutput)

# Add bias connections to our network
network.addConnection(biasToHidden)
network.addConnection(biasToOutput)

#initialize layers + modules are sorted topologically
network.sortModules()

# Print structure of our network
print(network)
# Print edges weight for connections between bias units and layers
print(biasToHidden.params)
print(biasToOutput.params)