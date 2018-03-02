import numpy as np
import numpy.random as random
from pybrain.datasets import ClassificationDataSet
from pybrain.structure.modules import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError
from sklearn import datasets


def splitWithProportion(self, proportion=0.7):
    """Produce two new datasets, the first one containing the fraction given
    by `proportion` of the samples."""
    indicies = random.permutation(len(self))
    separator = int(len(self) * proportion)

    leftIndicies = indicies[:separator]
    rightIndicies = indicies[separator:]

    leftDs = ClassificationDataSet(inp=self['input'][leftIndicies].copy(),
                                   target=self['target'][leftIndicies].copy())
    rightDs = ClassificationDataSet(inp=self['input'][rightIndicies].copy(),
                                    target=self['target'][rightIndicies].copy())
    return leftDs, rightDs


# Load iris data
irisData = datasets.load_iris()
dataFeatures = irisData.data
dataTargets = irisData.target


# Create data set object
dataSet = ClassificationDataSet(4, 1, nb_classes=3) # 3 - classes of iris

# Add data to out data set
for i in range(len(dataFeatures)):
    dataSet.addSample(np.ravel(dataFeatures[i]), dataTargets[i])

# Split data in train and test sets
trainingData, testData = splitWithProportion(dataSet, 0.7)

# Convert data classes to (1,0,0), (0,1,0), (0,0,1)
trainingData._convertToOneOfMany()
testData._convertToOneOfMany()

# Build neural network
neuralNetwork = buildNetwork(trainingData.indim, 7, trainingData.outdim, outclass=SoftmaxLayer)
trainer = BackpropTrainer(neuralNetwork, dataset=trainingData, momentum=0.01, learningrate=0.05, verbose=True)

# Train for 10 000 iterations and print error
trainer.trainEpochs(10000)
print('Error (test dataset): ', percentError(trainer.testOnClassData(dataset=testData), testData['class']))

# Iterate through and print out predictions
print('\n\n')
counter = 0
for input in dataFeatures:
    print(counter, " output is according to the NN: ", neuralNetwork.activate(input))
    counter = counter + 1