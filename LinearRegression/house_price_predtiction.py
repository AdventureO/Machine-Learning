import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("linreg.csv")
npMatrix = np.matrix(dataset)
X, Y = npMatrix[:,0], npMatrix[:,1]

regression = linear_model.LinearRegression()
regression.fit(X, Y)

a = regression.intercept_
b = regression.coef_[0]

print(X)
print(Y)

plt.scatter([X], [Y], color='black')
plt.plot(X, regression.predict(X), color='blue', linewidth=3)
plt.xlabel("x", fontsize=15)
plt.ylabel("Y", fontsize=15)
plt.show()

