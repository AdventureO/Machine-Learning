import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

#Read data from csv file
dataset = pd.read_csv("linreg.csv")
npMatrix = np.matrix(dataset)

#Prepared data sets
X, Y = npMatrix[:,0], npMatrix[:,1]

#Create linear regression model
regression = linear_model.LinearRegression()
#Train model with prepared data
regression.fit(X, Y)

#Line intersection with y label
a = regression.intercept_
#Slope of the line
b = regression.coef_[0]

#Plot our data and line we got after creating linear regression model
plt.scatter([X], [Y], color='black')
plt.plot(X, regression.predict(X), color='blue', linewidth=3)
plt.xlabel("x", fontsize=15)
plt.ylabel("Y", fontsize=15)
plt.show()