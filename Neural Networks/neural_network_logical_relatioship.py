from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

# Create network
neural_network = buildNetwork(2, 3, 1, bias=True)

#Create data set
data_set = SupervisedDataSet(2, 1)

# XOR logical relation data
# dataset.addSample((0, 0), (0,))
# dataset.addSample((0, 1), (1,))
# dataset.addSample((1, 0), (1,))
# dataset.addSample((1, 1), (0,))

# AND logical relation data
data_set.addSample((0, 0), (0,))
data_set.addSample((0, 1), (0,))
data_set.addSample((1, 0), (0,))
data_set.addSample((1, 1), (1,))

# Train via backpropagation
trainer = BackpropTrainer(neural_network, dataset=data_set, learningrate=0.01, momentum=0.06)

# Print put in the loop how error for our neural network is decreasing
for i in range(1, 30000):
    error = trainer.train()

    if i % 1000 == 0:
        print("Error in iteration ", i, " is: ", error)
        print(neural_network.activate([0, 0]))
        print(neural_network.activate([1, 0]))
        print(neural_network.activate([0, 1]))
        print(neural_network.activate([1, 1]))

print("\nFINAL SOLUTION\n")
print(neural_network.activate([0, 0]))
print(neural_network.activate([1, 0]))
print(neural_network.activate([0, 1]))
print(neural_network.activate([1, 1]))