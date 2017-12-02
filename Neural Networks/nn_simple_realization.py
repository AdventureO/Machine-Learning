import numpy as np

#Weigth of first layer
w1 = np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6]])

#Weight of second layer
w2 = np.zeros((1, 3))
w2[0,:] = np.array([0.5, 0.5, 0.5])

#Bias for first and second layer
b1 = np.array([0.8, 0.8, 0.8])
b2 = np.array([0.2])

#Activation function - sigmoid function
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def forward_propagation(n_layers, x, w, b):
    for l in range(n_layers - 1):
        if l == 0:
            node_in = x
        else:
            node_in = h
        #Input array for nodes in layer l + 1
        h = np.zeros((w[l].shape[0],))
        #Loop through each weight
        for i in range(w[l].shape[0]):
            #sum of activation function
            f_sum = 0
            #Loop through columns of weight array
            for j in range(w[l].shape[1]):
                f_sum += w[l][i][j] * node_in[j]
            #Add bias
            f_sum += b[l][i]
            #Use activation function to calculate h1, h2, h3
            h[i] = sigmoid_function(f_sum)
    return h

def matrix_forward_propagation(n_layers, x, w, b):
    for l in range(n_layers-1):
        if l == 0:
            node_in = x
        else:
            node_in = h
        z = w[l].dot(node_in) + b[l]
        h = sigmoid_function(z)
    return h

w = [w1, w2]
b = [b1, b2]
x = [1.5, 2.0, 3.0]

result = forward_propagation(3, x, w, b)
print(result)
result_1 = matrix_forward_propagation(3, x, w, b)
print(result_1)
