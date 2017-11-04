from pybrain.datasets import SupervisedDataSet

# Create dataset
data = SupervisedDataSet(2, 1)

# Add samples to data set we created
data.addSample((0, 0), (0))
data.addSample((1, 0), (1))
data.addSample((0, 1), (1))
data.addSample((1, 1), (0))

print(len(data))

# Print out data we created
for input, target in data:
    print("Input: ", input, " output: ", target)

print(data['input'])
print(data['target'])

