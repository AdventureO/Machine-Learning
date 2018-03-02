import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

# class 1
x1 = np.array([0,0.6,1.1,1.5,1.8,2.5,3,3.1,3.9,4,4.9,5,5.1])
y1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])

# class 2
x2 = np.array([3,3.8,4.4,5.2,5.5,6.5,6,6.1,6.9,7,7.9,8,8.1])
y2 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])

# prepared data for fiting model
X = np.array([[0],[0.6],[1.1],[1.5],[1.8],[2.5],[3],[3.1],[3.9],[4],[4.9],[5],[5.1],
              [3],[3.8],[4.4],[5.2],[5.5],[6.5],[6],[6.1],[6.9],[7],[7.9],[8],[8.1]])
y = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1])

# plot data (dots)
plt.plot(x1,y1,'ro',color='blue')
plt.plot(x2,y2,'ro',color='red')

# create logistic regression model
classifier = LogisticRegression()
# fit model with prepared data(1 feature)
classifier.fit(X,y)

# probability of belongings to classes(0 or 1) for 3
prediction = classifier.predict_proba(3)
print(prediction)

# return probability of belongings to some class of x
def logistic_function(classifier, x):
    return 1/(1 + np.exp(-(classifier.intercept_ + classifier.coef_ * x)))

# plot logistic function
for i in range(1, 120):
    plt.plot(i/10.0 - 1, logistic_function(classifier, i/10.0 - 1), 'ro', color='green')

plt.axis([-1, 10, -0.5, 1.5])
plt.show()