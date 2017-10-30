import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Blue points
xBlue = np.array([0.3,0.5,1,1.4,1.7,2])
yBlue = np.array([1,4.5,2.3,1.9,8.9,4.1])

# Red points
xRed = np.array([3.3,3.5,4,4.4,5.7,6])
yRed = np.array([7,1.5,6.3,1.9,2.9,7.1])

# Prepared data
X = np.array([[0.3,1],[0.5,4.5],[1,2.3],[1.4,1.9],[1.7,8.9],[2,4.1],[3.3,7],[3.5,1.5],[4,6.3],[4.4,1.9],[5.7,2.9],[6,7.1]])
Y = np.array([0,0,0,0,0,0,1,1,1,1,1,1]) # 0: blue class, 1: red class

# Plotting dots
plt.plot(xBlue, yBlue, 'ro', color = 'blue')
plt.plot(xRed, yRed, 'ro', color='red')

# The point for which we must predict the class
point = [1.1, 5]
plt.plot(point[0], point[1], 'ro', color='green', markersize=15)

# Measure of x,y plot
plt.axis([-0.5, 10, -0.5, 10])

# Create random forest classifier
classifierRF = RandomForestClassifier(n_estimators = 100) # number of decision trees
classifierRF.fit(X, Y)

# Predict class for our point
prediction = classifierRF.predict([point])
print(prediction)

# Create decision tree classifier
classifierDT = DecisionTreeClassifier()
classifierDT.fit(X, Y)

# Predict class for our point
prediction = classifierDT.predict([point])
print(prediction)


plt.show()
