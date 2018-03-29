import numpy as np
import scipy.signal as sig


data = np.array([[0, 105, 0], [40, 255, 90], [0, 55, 0]])
G_x = sig.convolve2d(data, np.array([[-1], [0], [1]]), mode='valid')
G_y = sig.convolve2d(data, np.array([[-1, 0, 1]]), mode='valid')
print(G_x, G_y)